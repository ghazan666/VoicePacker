# VoicePacker

基于 [FunASR](https://github.com/modelscope/FunASR) 的人声切片工具。

## 依赖

需要 [FFmpeg](https://ffmpeg.org/download.html)。
推荐用 [uv](https://docs.astral.sh/uv/getting-started/installation) 安装 Python 依赖：

```powershell
uv sync
```

如果有 NVIDIA 显卡，改为执行这条命令：

```powershell
uv sync --no-default-groups --group cu130
```

## 转写

把本地音频或 B 站视频转成 16 kHz 单声道 WAV，再识别人声，写出带时间戳和 sha 的文本。首次转写会从 ModelScope 下载约 1 GB 模型。
`--speaker` 会给每段加上说话人标签。

```powershell
uv run main.py transcribe "https://www.bilibili.com/video/BVxxxxxxxx"
uv run main.py transcribe "D:\path\to\audio.m4a" --speaker
```

默认输出到 `output/`：同名 `.wav` 和 `.txt`。

```text
[00:00:00.500 --> 00:00:04.200] [sha] 识别出的文字
```

## 切片

从转写生成的 `.txt` 里取关键词或 `[sha]`，在同一目录的同名 `.wav` 上切片。

```powershell
uv run main.py clip "要查找的文字"
uv run main.py clip "12位sha"
```

切片写入 `output/clips/`。

## 合并

合并多个语音切片。

```powershell
uv run main.py merge "D:\audio\1.wav" "D:\audio\2.wav" "D:\audio\3.wav"
```

结果写入 `<output>/merged/`。


## 上传

欢迎把生成的张云杰语音切片上传到这个网盘地址，建议先开自己的分支。游客账号已开放新建文件夹和上传权限。
```
https://alist.venetianfuntimecfwnfejikqklfwf.top:4443
```
