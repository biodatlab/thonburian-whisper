# Thonburian Whisper Longform Audio Transcription Tool

This command-line tool allows users to transcribe long-form Thai audio or video files using a Thonburian Whisper ASR (Automatic Speech Recognition) model. The transcriptions can be saved either as SubRip (SRT) subtitle files or as CSV files. Additionally, if you're transcribing a video, you have the option to burn the generated subtitles directly onto the video.

## Dependencies

### Requirements

Ensure you have `ffmpeg` installed on your system as it's used internally for audio and video processing:

```bash
sudo apt-get install ffmpeg
```

For Python dependencies, you can install the required Python libraries using `pip`

```
pip install -r requirements.txt
```

**Note** that youu might need to install `pytorch` with cuda 11.8 according to the official documentation.

## Usage

### Notebooks

You can refer to `thonburian_whisper_longform_notebook.ipynb` for a quick long form transcription.

### Command-line Script

The basic usage:

```bash
python main.py --input_file INPUT_FILE_PATH --output_file OUTPUT_FILE_PATH
```

**Arguments**

- `--input_file`: (Required) The path to your input video or audio file. Supported formats are `.mp4` for videos and `.wav` or `.mp3` for audios.
- `--output_file`: (Required) The path where you'd like to save your transcriptions. The format (CSV or SRT) will be determined by the `--output_format` argument.
- `--model_path`: The path to your Whisper ASR model. Defaults to a placeholder path (`/path/to/default/model`), so make sure to replace it with your actual model path.
- `--output_format`: The desired output format for your transcriptions. Choices are `csv` and `srt`, with `csv` being the default.
- `--burn_srt`: If this flag is provided and if the input is a video file, the generated SRT subtitles will be burned onto the video.

**Examples**

1. Video to SRT transcription

```bash
python main.py --input_file /path/to/video.mp4 --output_file /path/to/output.csv --output_format srt --model_path /path/to/model
```

2. Video transcription with burned subtitles

```bash
python main.py --input_file /path/to/video.mp4 --output_file /path/to/output.srt --output_format srt --model_path /path/to/model --burn_srt
```

3. Audio to CSV transcription

```bash
python main.py --input_file /path/to/audio.wav --output_file /path/to/output.csv --model_path /path/to/model
```
