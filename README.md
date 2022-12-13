# Thonburian Whisper

[Model](https://huggingface.co/biodatlab/whisper-th-medium-combined) | [Notebook](https://github.com/biodatlab/whisper-th-demo/blob/main/whisper_th_demo.ipynb) | [Hugginface Space](https://huggingface.co/spaces/biodatlab/whisper-thai-demo) | Blog (TBA)

<img src="assets/Thonburian-Whisper-1.jpg" width="500"/>

Thonburian Whisper is an Automatic Speech Recognition (ASR) in Thai fine-tuned using [Whisper](https://openai.com/blog/whisper/) model
originally from OpenAI.

## Usage

Use the model with [Hugginface's transformers](https://github.com/huggingface/transformers) as follows:

```py
import torch
from transformers import pipeline

MODEL_NAME = "biodatlab/whisper-th-medium-combined"
lang = "th"

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
```

## Requirements

Use `pip` to install the requirements as follows:

``` sh
!pip install git+https://github.com/huggingface/transformers
!pip install librosa
!sudo apt install ffmpeg
```
