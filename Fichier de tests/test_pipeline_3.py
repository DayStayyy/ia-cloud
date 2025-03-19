# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install transformers
# pip install datasets
# pip install librosa
# pip install transformers
# pip3 install ffmpeg-python
# necessite ffmepg installé. Sous windows, utiliser winget install ffmpeg et reboot (ou wsl)

import torch
from transformers import pipeline
import ffmpeg
import math

# extract audio from the input video
input_video = "input.mp4"
input_video_name = input_video.replace(".mp4", "")
extracted_audio = f"audio-{input_video_name}.wav"
stream = ffmpeg.input(input_video)
stream = ffmpeg.output(stream, extracted_audio)
ffmpeg.run(stream, overwrite_output=True)

device = "cpu"
whisper = pipeline("automatic-speech-recognition",
                   "openai/whisper-small.en",
                   torch_dtype=torch.float16,
                   device=device)

transcription = whisper(extracted_audio, chunk_length_s=28, return_timestamps=True)

def format_time(seconds):

    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"

    return formatted_time


subtitle_file = f"sub-{input_video_name}.en.srt"
text = ""
offset=0
for index, chunk in enumerate(transcription["chunks"]):
    start = offset + chunk["timestamp"][0]
    end = offset + chunk["timestamp"][1]
    if start > end:
        # chunck is the delimitation of a new segment
        offset += 28
        continue
    text_chunk = chunk["text"]
    segment_start = format_time(start)
    segment_end = format_time(end)
    text += f"{str(index + 1)} \n"
    text += f"{segment_start} --> {segment_end} \n"
    text += f"{text_chunk} \n"
    text += "\n"

f = open(subtitle_file, "w")
f.write(text)
f.close()


video_input_stream = ffmpeg.input(input_video)
subtitle_input_stream = ffmpeg.input(subtitle_file)
output_video = f"output.mp4"
subtitle_track_title = subtitle_file.replace(".srt", "")

# Change if subtitile must be umbedded or not in the video
embedded_subtitle = False
if not embedded_subtitle:
    stream = ffmpeg.output(
        video_input_stream, subtitle_input_stream, output_video, **{"c": "copy", "c:s": "mov_text"},
        **{"metadata:s:s:0": "language=en",
        "metadata:s:s:1": f"title={subtitle_track_title}"}
    )
    ffmpeg.run(stream, overwrite_output=True)
else:
    stream = ffmpeg.output(video_input_stream, output_video,
                           vf=f"subtitles={subtitle_file}")

    ffmpeg.run(stream, overwrite_output=True)

print(transcription["text"])
