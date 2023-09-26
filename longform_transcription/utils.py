import math
import json
import re
import os
from datetime import timedelta
import torch
from attacut import tokenize
from collections import Counter


SAMPLING_RATE = 16000


def perform_vad(src: str, dest_dir: str) -> tuple:
    """
    Perform Voice Activity Detection (VAD) on the given audio source file
    and save resulting audio chunks to the destination directory.

    Parameters:
    - src (str): Path to the source audio file.
    - dest_dir (str): Path to the destination directory where chunks will be saved.

    Returns:
    - tuple: Tuple containing wav data and chunklist information.
    """

    # Try to load the VAD model and utilities
    try:
        smodel, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=False,
        )
        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = utils
    except Exception as e:
        raise RuntimeError(f"Failed to load silero-vad model and utilities. Error: {e}")

    # Try to read the audio
    try:
        wav = read_audio(src, sampling_rate=SAMPLING_RATE)
    except Exception as e:
        raise RuntimeError(f"Failed to read audio from {src}. Error: {e}")

    # Get speech timestamps
    st = get_speech_timestamps(
        wav,
        smodel,
        threshold=0.65,
        sampling_rate=SAMPLING_RATE,
        min_speech_duration_ms=500,
        min_silence_duration_ms=100,
        window_size_samples=1536,
        return_seconds=False,
    )

    total_samples = list(wav.size())[0]
    chunklist = []

    # Process the speech timestamps
    for i, s in enumerate(st):
        fname = os.path.join(
            dest_dir, os.path.splitext(os.path.basename(src))[0] + f"_{i:05d}.wav"
        )

        start = s["start"] - int(120 * SAMPLING_RATE / 1000)
        start = max(start, 0)

        end = s["end"] + int(60 * SAMPLING_RATE / 1000)
        end = min(end, total_samples - 1)

        chunklist.append(
            {"start": start, "end": end, "idx": i, "text": "", "fname": fname}
        )

    tempchunk = os.path.join(
        dest_dir, os.path.splitext(os.path.basename(src))[0] + "_chunk.json"
    )

    # Conditionally create the folder if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Save the chunk information to a json file
    try:
        with open(tempchunk, "w", encoding="utf-8") as fp:
            json.dump(chunklist, fp)
    except Exception as e:
        raise RuntimeError(
            f"Failed to save chunk information to {tempchunk}. Error: {e}"
        )

    # Save audio chunks to the destination directory
    for c in chunklist:
        try:
            save_audio(
                c["fname"],
                collect_chunks([c], wav),
                sampling_rate=SAMPLING_RATE,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to save audio chunk to {c['fname']}. Error: {e}"
            )

    return wav, chunklist


def max_intervals(range_length: float, min_interval_length: float) -> int:
    """
    Calculate the maximum number of intervals that fit within a given range length.

    Parameters:
        range_length (float): Total length of the range.
        min_interval_length (float): Desired minimum interval length.

    Returns:
        int: Maximum number of intervals.
    """
    return int(math.ceil(range_length / min_interval_length))


def sum_index_list(numbers: list[int]) -> list[int]:
    """
    Compute the cumulative sum of numbers in a list.

    Parameters:
        numbers (list[int]): List of numbers.

    Returns:
        list[int]: List with cumulative sums.
    """
    if not numbers:
        return []

    result = [numbers[0]]
    for i in range(1, len(numbers)):
        result.append(numbers[i] + result[i - 1])

    return result


def is_english(text: str) -> bool:
    """
    Check if the given text is in English.

    Parameters:
        text (str): Text to check.

    Returns:
        bool: True if the text is in English, False otherwise.
    """
    return bool(re.match(r"[a-zA-Z]", text)) or text == " "


def convert_mp4_to_wav(video_file_path: str) -> str:
    """
    Convert an MP4 video file to WAV audio format.

    Parameters:
        video_file_path (str): Path to the MP4 file.

    Returns:
        str: Path to the generated WAV file.
    """
    _, file_extension = os.path.splitext(video_file_path)
    os.system(
        f'ffmpeg -i "{video_file_path}" -y -ar 16000 -ac 1 -c:a pcm_s16le "{video_file_path.replace(file_extension, ".wav")}"'
    )
    return video_file_path.replace(file_extension, ".wav")


def duration_check(
    millis_times: tuple,
    time_concat: list,
    required_duration: int,
    sampling_rate: int = 16000,
) -> tuple:
    """
    Check if the duration of the current segment exceeds a required duration.

    Parameters:
        millis_times (tuple): Start and end times of the segment.
        time_concat (list): List of segments.
        required_duration (int): Desired segment duration.
        sampling_rate (int, optional): Sampling rate of the audio. Defaults to 16000.

    Returns:
        tuple: Updated list of segments and a boolean indicating if the desired duration is exceeded.
    """
    time_concat.append(millis_times)
    duration = int((time_concat[-1][1] - time_concat[0][0]) / sampling_rate)
    return time_concat, duration >= required_duration


def postprocess_text(txt: str, limit: int = 5) -> str:
    """
    Reduce repeated occurrences of words in a text based on a set limit.

    Parameters:
        txt (str): Text to process.
        limit (int, optional): Maximum allowed repetitions for any word. Defaults to 5.

    Returns:
        str: Processed text.
    """
    space_count = Counter(txt)[" "]

    # Tokenize based on space count.
    if space_count > 10:
        words = txt.split(" ")
    else:
        words = tokenize(txt)

    word_counts = Counter(words)
    return " ".join([word for word in words if word_counts[word] <= limit])


from datetime import timedelta
import os


def generate_srt(transcriptions: list, srt_filename: str) -> None:
    """
    Generate an SRT file based on the provided transcriptions.

    Parameters:
    - transcriptions (list): A list of transcription chunks with start and end timestamps and text.
    - srt_filename (str): Path to save the generated SRT file.
    """
    try:
        # Open the file once to write all segments
        with open(srt_filename, "w", encoding="utf-8") as srtfile:
            for idx, chunk in enumerate(transcriptions):
                timestamp = (chunk["start"], chunk["end"])
                starttime = str(0) + str(timedelta(seconds=int(timestamp[0]))) + ",000"
                endtime = str(0) + str(timedelta(seconds=int(timestamp[1]))) + ",000"
                text = chunk["text"]
                text = "inaudible" if text.strip() == "" else text
                segment_id = idx + 1
                segment = f"{segment_id}\n{starttime} --> {endtime}\n{text}\n\n"
                srtfile.write(segment)
    except Exception as e:
        raise RuntimeError(f"Failed to generate the SRT file. Error: {e}")


def burn_srt_to_video(video_in: str, srt_file: str) -> str:
    """
    Burn the provided SRT file to the given video.

    Parameters:
    - video_in (str): Path to the input video.
    - srt_file (str): Path to the SRT file to be burned onto the video.

    Returns:
    - str: Path to the output video with subtitles.
    """
    print("Starting creation of video with srt")

    filename = os.path.basename(video_in).split(".")[0]
    video_out = filename + "_with_subtitles.mp4"

    try:
        command = '/usr/bin/ffmpeg -i "{}" -y -vf subtitles="{}" "{}"'.format(
            video_in, srt_file, video_out
        )
        os.system(command)
    except Exception as e:
        print(f"Failed to burn subtitles to video. Error: {e}")

    return video_out
