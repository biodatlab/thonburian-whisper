<p align="center">
  <img src="assets/thonburian-whisper-logo.png" width="400"/>
</p>

[🤖 Model](https://huggingface.co/biodatlab/whisper-th-medium-combined) | [📔 Jupyter Notebook](https://github.com/biodatlab/thonburian-whisper/blob/main/thonburian_whisper_notebook.ipynb) | [🤗 Huggingface Space Demo](https://huggingface.co/spaces/biodatlab/whisper-thai-demo) | [📃 Medium Blog (Thai)](https://medium.com/@Loolootech/thonburian-whisper-asr-27c067c534cb)

**Thonburian Whisper** is an Automatic Speech Recognition (ASR) model for Thai, fine-tuned using [Whisper](https://openai.com/blog/whisper/) model
originally from OpenAI. The model is released as a part of Huggingface's [Whisper fine-tuning event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)  (December 2022). We trained the model using [Commonvoice](https://commonvoice.mozilla.org/th) 13, [Gowajee corpus](https://github.com/ekapolc/gowajee_corpus), and [Thai Elderly Speech dataset](https://github.com/VISAI-DATAWOW/Thai-Elderly-Speech-dataset/releases/tag/v1.0.0) datasets. Our model demonstrates robustness under environmental noise and fine-tuned abilities to
domain-specific audio such as financial and medical domains.

## Usage

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biodatlab/thonburian-whisper/blob/main/thonburian_whisper_notebook.ipynb)

Use the model with [Huggingface's transformers](https://github.com/huggingface/transformers) as follows:

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

# Perform ASR with the created pipe.
pipe("audio.mp3", generate_kwargs={"language":"<|th|>", "task":"transcribe"}, batch_size=16)["text"]
```

## Requirements

Use `pip` to install the requirements as follows:

```sh
!pip install git+https://github.com/huggingface/transformers
!pip install librosa
!sudo apt install ffmpeg
```

## Model performance

We measure word error rate (WER) of the model with [deepcut tokenizer](https://github.com/rkcosmos/deepcut) after
normalizing special tokens (▁ to _ and — to -) and simple text-postprocessing (เเ to แ and  ํา to  ำ).

| **Model**                         | **WER (Commonvoice 13)** |
| --------------------------------- | ------------------------ |
| Thonburian Whisper (small) [Link](https://huggingface.co/biodatlab/whisper-th-small-combined)      | **13.14**                 |
| Thonburian Whisper (medium) [Link](https://huggingface.co/biodatlab/whisper-th-medium-combined)       | **7.42**                 |
| Thonburian Whisper (large-v2) [Link](https://huggingface.co/biodatlab/whisper-th-large-combined)     | **7.69**                 |
| Thonburian Whisper (large-v3) [Link](https://huggingface.co/biodatlab/whisper-th-large-v3-combined)      | **6.59**                 |


Thonburian Whisper is fine-tuned with a combined dataset of Thai speech including common voice, google fleurs, and curated datasets.
The common voice test splitting is based on original splitting from [`datasets`](https://huggingface.co/docs/datasets/index).

**Inference time**

We have performed benchmark average inference speed on 1 minute audio with different model sizes (small, medium, and large)
on NVIDIA A100 with 32 fp, batch size of 32. The medium model presents a balanced trade-off between WER and computational costs.

| **Model**                   | **Memory usage (Mb)**    | **Inference time (sec / 1 min)** | **Number of Parameters** |
| --------------------------- | ------------------------ | -------------------------------- | ------------------------ |
| Thonburian Whisper (small) [Link](https://huggingface.co/biodatlab/whisper-th-small-combined)  | 7,194                    | 4.83                             | 242M                     |
| Thonburian Whisper (medium) [Link](https://huggingface.co/biodatlab/whisper-th-medium-combined) | 10,878                   | 7.11                             | 764M                     |
| Thonburian Whisper (large) [Link](https://huggingface.co/biodatlab/whisper-th-large-combined)  | 18,246                   | 9.61                             | 1540M                    |
| Distilled Thonburian Whisper (small) [Link](https://huggingface.co/biodatlab/distill-whisper-th-small) | 4,944            |              TBA                    | 166M                     |
| Distilled Thonburian Whisper (medium) [Link](https://huggingface.co/biodatlab/distill-whisper-th-medium) | 7,084           |               TBA                   | 428M                     |

## Long-form Inference

Thonburian Whisper can be used for long-form audio transcription by combining VAD, Thai-word tokenizer, and chunking for word-level alignment.
We found that this is more robust and produce less insertion error rate (IER) comparing to using Whisper with timestamp. See `README.md` in [longform_transcription](https://github.com/biodatlab/thonburian-whisper/tree/main/longform_transcription) folder for detail usage.


## Developers

- [Biomedical and Data Lab, Mahidol University](https://biodatlab.github.io/)
- [Looloo technology](https://loolootech.com/)

## Citation

If you use the model, you can cite it with the following bibtex.

```
@misc {thonburian_whisper_med,
    author       = { Zaw Htet Aung, Thanachot Thavornmongkol, Atirut Boribalburephan, Vittavas Tangsriworakan, Knot Pipatsrisawat, Titipat Achakulvisut },
    title        = { Thonburian Whisper: A fine-tuned Whisper model for Thai automatic speech recognition },
    year         = 2022,
    url          = { https://huggingface.co/biodatlab/whisper-th-medium-combined },
    doi          = { 10.57967/hf/0226 },
    publisher    = { Hugging Face }
}
```
