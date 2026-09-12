import argparse
import bisect
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import torch
import yt_dlp
from funasr import AutoModel

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
CLIPS_DIR = OUTPUT_DIR / "clips"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SHA_HEX_LENGTH = 12
SHA_QUERY = re.compile(rf"^[0-9a-fA-F]{{{SHA_HEX_LENGTH},64}}$")
LOUDNORM_I = -16.0
LOUDNORM_TP = -1.5
LOUDNORM_LRA = 11.0
LOUDNORM_JSON = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)
DEFAULT_VAD_KWARGS = {
    "max_single_segment_time": 30000,
    "silero_threshold": 0.35,
    "silero_min_speech_duration_ms": 250,
    "silero_min_silence_duration_ms": 350,
    "silero_speech_pad_ms": 120,
}
BILIBILI_URL = re.compile(r"(?i)^(?:https?://)?(?:(?:www|m|live|space)\.)?(?:bilibili\.com|b23\.tv)(?:[:/?]|$)")
TIMESTAMP_LINE = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2}\.\d{3}) --> "
    r"(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})\]\s*"
    r"(?:(?P<speaker>Speaker\d+|Unknown Speaker):\s*)?"
    r"(?:\[(?P<sha>[0-9a-fA-F]{8,64})\]\s*)?"
    r"(?P<text>.*)$"
)


@dataclass(frozen=True)
class TranscriptSegment:
    start: str
    end: str
    text: str
    speaker: str | None = None
    sha: str | None = None


def require_ffmpeg(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("请检查是否已安装 FFmpeg，并确保其在 PATH 中。")
        return func(*args, **kwargs)

    return wrapper


def resolve_project_directory(path: Path | None, default: Path) -> Path:
    relative_path = default if path is None else path
    if relative_path.is_absolute():
        raise ValueError(f"目录必须使用相对路径: {relative_path}")

    resolved_path = (PROJECT_DIR / relative_path).resolve()
    if resolved_path != PROJECT_DIR and PROJECT_DIR not in resolved_path.parents:
        raise ValueError(f"目录不能超出项目范围: {relative_path}")
    return resolved_path


def format_timestamp(milliseconds: int) -> str:
    total_seconds, millis = divmod(milliseconds, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def parse_timestamp(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def safe_filename(text: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:100].rstrip(" .") or "clip"


def is_bilibili_url(value: str) -> bool:
    return bool(BILIBILI_URL.match(value.strip()))


def download_bilibili_audio(url: str, output_dir: Path) -> Path:
    before = {path.resolve() for path in output_dir.iterdir() if path.is_file()}
    options = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "overwrites": True,
        "windowsfilenames": True,
        "outtmpl": str(output_dir / "%(title).80B [%(id)s].%(ext)s"),
        "quiet": False,
        "noprogress": False,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url.strip(), download=True)
            if info and info.get("entries"):
                info = next(
                    (entry for entry in info["entries"] if entry),
                    None,
                )
            requested = (info or {}).get("requested_downloads") or []
            filename = None
            if requested and requested[0].get("filepath"):
                filename = requested[0]["filepath"]
            elif info:
                filename = downloader.prepare_filename(info)
    except yt_dlp.utils.DownloadError as error:
        raise RuntimeError(f"B 站音频下载失败: {error}") from error

    audio_path = Path(filename) if filename else None
    if audio_path is None or not audio_path.is_file():
        after = [path for path in output_dir.iterdir() if path.is_file() and path.resolve() not in before]
        if not after:
            raise RuntimeError("B 站音频下载失败：未找到下载文件。")
        audio_path = max(after, key=lambda path: path.stat().st_mtime)
    print(f"Downloaded: {audio_path}")
    return audio_path


def segment_sha(start_ms: int, end_ms: int, text: str) -> str:
    value = f"{start_ms}:{end_ms}:{text}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:SHA_HEX_LENGTH]


@require_ffmpeg
def convert_to_wav(audio_path: Path, output_dir: Path) -> Path:
    wav_path = output_dir / f"{audio_path.stem}.wav"

    if audio_path == wav_path and audio_path.suffix.casefold() == ".wav":
        return wav_path

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        check=True,
    )

    audio_path.unlink()
    print(f"Deleted source: {audio_path}")
    return wav_path


def format_segments(segments: list[dict], include_speaker: bool) -> list[str]:
    lines = []
    for segment in segments:
        start = format_timestamp(int(segment["start"]))
        end = format_timestamp(int(segment["end"]))
        prefix = f"[{start} --> {end}]"
        if include_speaker:
            speaker_id = segment.get("spk")
            speaker = f"Speaker{int(speaker_id)}" if speaker_id is not None else "Unknown Speaker"
            prefix += f" {speaker}:"

        text = (segment.get("sentence") or segment.get("text", "")).strip()
        if not text:
            print(
                f"Warning: skipped empty segment at {start}.",
                file=sys.stderr,
            )
            continue
        sha = segment_sha(int(segment["start"]), int(segment["end"]), text)
        lines.append(f"{prefix} [{sha}] {text}")
    return lines


def transcribe_audio(
    wav_path: Path,
    distinguish_speakers: bool = False,
) -> list[str]:
    model_options = dict(
        model="iic/SenseVoiceSmall",
        vad_model="silero-vad",
        vad_kwargs=dict(DEFAULT_VAD_KWARGS),
        device=DEVICE,
        disable_update=True,
    )
    if distinguish_speakers:
        model_options.update(spk_model="cam++")
    model = AutoModel(**model_options)
    result = model.generate(
        input=str(wav_path),
        batch_size_s=300,
        sentence_timestamp=True,
    )

    segments = result[0].get("sentence_info", [])
    if not segments:
        raise RuntimeError("FunASR 未返回带时间戳的片段。")
    return format_segments(
        segments,
        include_speaker=distinguish_speakers,
    )


def transcribe_file(
    audio: str | Path,
    distinguish_speakers: bool = False,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_input = str(audio).strip()
    if is_bilibili_url(audio_input):
        audio_path = download_bilibili_audio(audio_input, output_dir)
    else:
        audio_path = Path(audio_input).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    wav_path = convert_to_wav(audio_path, output_dir)
    print(f"WAV: {wav_path}")

    lines = transcribe_audio(wav_path, distinguish_speakers)
    transcript_path = output_dir / f"{wav_path.stem}.txt"
    transcript_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Transcript: {transcript_path}")
    return wav_path, transcript_path


def available_clip_path(text: str, clips_dir: Path = CLIPS_DIR) -> Path:
    clips_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(text)
    candidate = clips_dir / f"{name}.wav"
    suffix = 2
    while candidate.exists():
        candidate = clips_dir / f"{name}_{suffix}.wav"
        suffix += 1
    return candidate


def loudnorm_filter(measured: dict[str, str] | None = None) -> str:
    filter_spec = f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
    if measured is None:
        return f"{filter_spec}:print_format=json"
    return (
        f"{filter_spec}:"
        f"measured_I={measured['input_i']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:"
        f"linear=true"
    )


def _is_finite_loudness(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def measure_clip_loudness(wav_path: Path, start_seconds: float, duration: float) -> dict[str, str] | None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(wav_path),
            "-t",
            f"{duration:.3f}",
            "-af",
            loudnorm_filter(),
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = LOUDNORM_JSON.search(result.stderr)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    required = (
        "input_i",
        "input_tp",
        "input_lra",
        "input_thresh",
        "target_offset",
    )
    if any(not _is_finite_loudness(data.get(key)) for key in required):
        return None
    return {key: str(data[key]) for key in required}


def export_normalized_clip(
    wav_path: Path,
    start_seconds: float,
    duration: float,
    clip_path: Path,
) -> None:
    measured = measure_clip_loudness(wav_path, start_seconds, duration)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(wav_path),
            "-t",
            f"{duration:.3f}",
            "-af",
            loudnorm_filter(measured),
            "-c:a",
            "pcm_s16le",
            str(clip_path),
        ],
        check=True,
    )


def read_transcript(transcript_path: Path) -> list[TranscriptSegment]:
    segments = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        match = TIMESTAMP_LINE.match(line)
        if not match:
            continue
        segments.append(
            TranscriptSegment(
                start=match.group("start"),
                end=match.group("end"),
                text=match.group("text"),
                speaker=match.group("speaker"),
                sha=match.group("sha"),
            )
        )
    return segments


def find_transcript_matches(transcript_path: Path, query: str) -> list[tuple[str, str, str]]:
    if not query:
        raise ValueError("请输入要查找的文本。")

    segments = read_transcript(transcript_path)
    joined_text = ""
    total_length = 0
    boundaries = []

    for segment in segments:
        joined_text += segment.text
        total_length += len(segment.text)
        boundaries.append(total_length)

    matches = []
    seen_indexes: set[int] = set()
    search_from = 0
    while True:
        position = joined_text.find(query, search_from)
        if position < 0:
            break
        index = bisect.bisect_right(boundaries, position)
        search_from = position + len(query)
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        matches.append((segments[index].start, segments[index].end, segments[index].text))
    return matches


def find_sha_matches(sha: str, search_dir: Path) -> list[tuple[Path, str, str, str]]:
    query = sha.casefold()
    matches = []
    for transcript_path in sorted(search_dir.glob("*.txt")):
        wav_path = transcript_path.with_suffix(".wav")
        if not wav_path.is_file():
            continue
        for segment in read_transcript(transcript_path):
            if segment.sha and segment.sha.casefold().startswith(query):
                matches.append((wav_path, segment.start, segment.end, segment.text))
    return matches


@require_ffmpeg
def clip_by_text(query: str, search_dir: Path = OUTPUT_DIR) -> Path:
    if not search_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {search_dir}")

    matches: list[tuple[Path, str, str, str]] = []
    if SHA_QUERY.fullmatch(query):
        matches = find_sha_matches(query, search_dir)
        if len(matches) > 1:
            raise ValueError(f'SHA 前缀 "{query}" 匹配到 {len(matches)} 个片段，请提供更长的 SHA。')

    if not matches:
        for transcript_path in sorted(search_dir.glob("*.txt")):
            wav_path = transcript_path.with_suffix(".wav")
            if not wav_path.is_file():
                continue
            matches.extend(
                (wav_path, start, end, text) for start, end, text in find_transcript_matches(transcript_path, query)
            )

    if not matches:
        raise ValueError(f'输出目录没有包含 "{query}" 的转录片段。')

    clip_path: Path | None = None
    for wav_path, start, end, text in matches:
        clip_path = available_clip_path(text, search_dir / "clips")
        start_seconds = parse_timestamp(start)
        duration = parse_timestamp(end) - start_seconds
        export_normalized_clip(wav_path, start_seconds, duration, clip_path)
        print(f"[{start} --> {end}] {text}")
        print(f"Clip: {clip_path}")

    if clip_path is None:
        raise ValueError(f'输出目录没有包含 "{query}" 的转录片段。')
    return clip_path


def available_output_path(path: Path) -> Path:
    candidate = path
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        suffix += 1
    return candidate


@require_ffmpeg
def merge_audio_files(audio_paths: list[Path], output_dir: Path = OUTPUT_DIR) -> Path:
    if len(audio_paths) < 2:
        raise ValueError("至少需要两个音频文件才能合并。")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths = [path.resolve() for path in audio_paths]
    missing_paths = [path for path in resolved_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"所选音频文件不存在: {missing_paths[0]}")

    merged_name = safe_filename("_".join(path.stem for path in resolved_paths))
    merged_path = available_output_path(output_dir / f"{merged_name}.wav")

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for audio_path in resolved_paths:
        command.extend(["-i", str(audio_path)])
    filters = []
    for index in range(len(resolved_paths)):
        filters.append(f"[{index}:a]aresample=16000," f"aformat=sample_fmts=fltp:channel_layouts=mono[a{index}]")
    inputs = "".join(f"[a{index}]" for index in range(len(resolved_paths)))
    filters.append(f"{inputs}concat=n={len(resolved_paths)}:v=0:a=1[out]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(merged_path),
        ]
    )
    subprocess.run(command, check=True)
    print(f"Merged WAV: {merged_path}")
    return merged_path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="音频转录和剪辑工具")
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = commands.add_parser(
        "transcribe",
        help="转化音频为WAV并转录",
    )
    transcribe_parser.add_argument(
        "audio",
        help="本地音频文件路径，或 B 站视频链接",
    )
    transcribe_parser.add_argument(
        "--speaker",
        action="store_true",
        help="启用说话人区分",
    )
    transcribe_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="相对于项目根目录的输出目录（默认: output）",
    )

    clip_parser = commands.add_parser(
        "clip",
        help="输入文本，在输出的转录文本中查找包含该文本的片段，并创建WAV剪辑",
    )
    clip_parser.add_argument("text", help="查找包含该文本的转录片段，并创建WAV剪辑")
    clip_parser.add_argument(
        "--path",
        type=Path,
        help="相对于项目根目录的转录文件目录（默认: output）",
    )

    merge_parser = commands.add_parser(
        "merge",
        help="合并多个音频文件",
    )
    merge_parser.add_argument(
        "audio",
        nargs="+",
        type=Path,
        help="要合并的音频文件列表，按顺序排列",
    )
    merge_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="相对于项目根目录的输出目录（默认: output/merged）",
    )
    args = parser.parse_args()

    if args.command == "transcribe":
        transcribe_file(
            args.audio,
            distinguish_speakers=args.speaker,
            output_dir=resolve_project_directory(args.output, Path("output")),
        )
    elif args.command == "clip":
        clip_by_text(
            args.text,
            search_dir=resolve_project_directory(args.path, Path("output")),
        )
    elif args.command == "merge":
        merge_audio_files(
            args.audio,
            output_dir=resolve_project_directory(args.output, Path("output/merged")),
        )


if __name__ == "__main__":
    main()
