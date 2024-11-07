<p align="center">
  <img src="assets/thonburian-whisper-logo.png" width="400"/>
</p>

<p align="center">
  <img width="30px" src="assets/wordsense-looloo.png" />
  <img width="100px" src="assets/looloo-logo.png" />
</p>


[🤖 Model](https://huggingface.co/biodatlab/whisper-th-medium-combined) | [📔 Jupyter Notebook](https://github.com/biodatlab/thonburian-whisper/blob/main/thonburian_whisper_notebook.ipynb) | [🤗 Huggingface Space Demo](https://huggingface.co/spaces/biodatlab/whisper-thai-demo) | [📃 Medium Blog (Thai)](https://medium.com/@Loolootech/thonburian-whisper-asr-27c067c534cb) | [📄 Publication at ICNLSP](https://aclanthology.org/2024.icnlsp-1.17/)

**Thonburian Whisper** is an Automatic Speech Recognition (ASR) model for Thai, fine-tuned using [Whisper](https://openai.com/blog/whisper/) model
originally from OpenAI. The model is released as a part of Huggingface's [Whisper fine-tuning event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)  (December 2022). We fine-tuned Whisper models for Thai using [Commonvoice](https://commonvoice.mozilla.org/th) 13, [Gowajee corpus](https://github.com/ekapolc/gowajee_corpus), [Thai Elderly Speech](https://github.com/VISAI-DATAWOW/Thai-Elderly-Speech-dataset/releases/tag/v1.0.0), [Thai Dialect](https://github.com/SLSCU/thai-dialect-corpus) datasets. Our models demonstrate robustness under environmental noise and fine-tuned abilities to domain-specific audio such as financial and medical domains. We release models and distilled models on Huggingface model hubs (see below).

## Usage

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biodatlab/thonburian-whisper/blob/main/thonburian_whisper_notebook.ipynb)

Use the model with [Huggingface's transformers](https://github.com/huggingface/transformers) as follows:

```py
import torch
from transformers import pipeline

MODEL_NAME = "biodatlab/whisper-th-medium-combined"  # see alternative model names below
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

## Model checkpoint and performance

We measure word error rate (WER) of the model with [deepcut tokenizer](https://github.com/rkcosmos/deepcut) after
normalizing special tokens (▁ to _ and — to -) and simple text-postprocessing (เเ to แ and  ํา to  ำ). See an example evaluation script [here](https://github.com/biodatlab/thonburian-whisper/blob/main/evaluations/evaluate.py).

| Model                    | WER (Commonvoice 13) | Model URL |
|------------------------------|--------------------------|---------------|
| Thonburian Whisper (small)   | 11.0     | [Link](https://huggingface.co/biodatlab/whisper-th-small-combined) |
| Thonburian Whisper (medium)  | 7.42      | [Link](https://huggingface.co/biodatlab/whisper-th-medium-combined) |
| Thonburian Whisper (large-v2)| 7.69      | [Link](https://huggingface.co/biodatlab/whisper-th-large-combined) |
| Thonburian Whisper (large-v3)| 6.59      | [Link](https://huggingface.co/biodatlab/whisper-th-large-v3-combined) |
| Distilled Thonburian Whisper (small) | 11.2 | [Link](https://huggingface.co/biodatlab/distill-whisper-th-small) |
| Distilled Thonburian Whisper (medium) | 7.6 | [Link](https://huggingface.co/biodatlab/distill-whisper-th-medium) |
| Distilled Thonburian Whisper (large) | 6.82 | [Link](https://huggingface.co/biodatlab/distill-whisper-th-large-v3) |
| Thonburian Whisper (medium-timestamps) | 15.57 | [Link](https://huggingface.co/biodatlab/whisper-th-medium-timestamps) |


Thonburian Whisper is fine-tuned with a combined dataset of Thai speech including common voice, google fleurs, and curated datasets.
The common voice test splitting is based on original splitting from [`datasets`](https://huggingface.co/docs/datasets/index).

**Inference time**

We have performed benchmark average inference speed on 1 minute audio with different model sizes (small, medium, and large)
on NVIDIA A100 with fp32 precision, batch size of 1. The medium model presents a balanced trade-off between WER and computational costs. (Note that the distilled models due to their smaller size and a batch size of 1 are not fully saturating the GPU. With higher batch size, the inference time will be lower substantially.)

| Model                            | Memory usage (Mb) | Inference time (sec / 1 min) | Number of Parameters | Model URL |
|----------------------------------|-------------------|------------------------------|----------------------|-----------|
| Thonburian Whisper (small)           | 931.93       | 0.50               | 242M       | [Link](https://huggingface.co/biodatlab/whisper-th-small-combined) |
| Thonburian Whisper (medium)          | 2923.05      | 0.83                | 764M       | [Link](https://huggingface.co/biodatlab/whisper-th-medium-combined) |
| Thonburian Whisper (large)           | 6025.84      | 1.89                | 1540M      | [Link](https://huggingface.co/biodatlab/whisper-th-large-combined) |
| Distilled Thonburian Whisper (small) | 650.27       | 4.42                 | 166M       | [Link](https://huggingface.co/biodatlab/distill-whisper-th-small) |
| Distilled Thonburian Whisper (medium)| 1642.15       | 4.36                | 428M       | [Link](https://huggingface.co/biodatlab/distill-whisper-th-medium) |
| Distilled Thonburian Whisper (large) | 3120.05 | 5.5 | 809M | [Link](https://huggingface.co/biodatlab/distill-whisper-th-large-v3) |


## Model Types and Use Cases

### Thonburian Whisper (Standard Models)

These models are fine-tuned versions of OpenAI's Whisper, optimized for Thai ASR:

- **Small**: Balanced performance with lower resource requirements.
- **Medium**: Best trade-off between accuracy and computational cost.
- **Large-v2/v3**: Highest accuracy, but more resource-intensive.

Use these for general Thai ASR tasks where timestamps are not required.

### Thonburian Whisper with Timestamps

**Model**: `biodatlab/whisper-th-medium-timestamp`

This model is specifically designed for Thai ASR with timestamp generation. It's based on the Whisper medium architecture and fine-tuned on a custom longform dataset.

**Key Features**:
- Generates timestamps for transcribed text
- WER: 15.57 (with Deepcut Tokenizer)
- Suitable for subtitle creation or audio-text alignment tasks

**Usage**:

```python
from transformers import pipeline
import torch
MODEL_NAME = "biodatlab/whisper-th-medium-timestamp"
lang = "th"
device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
    return_timestamps=True,
)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
    language=lang,
    task="transcribe"
)
result = pipe("audio.mp3", return_timestamps=True)
text, timestamps = result["text"], result["chunks"]
```

**Note**: While this model provides timestamp information, its accuracy may be lower than non-timestamped versions due to several factors.

### Distilled Thonburian Whisper Models

These models are distilled versions of the larger Thonburian Whisper models, offering improved efficiency:

1. **Distilled Medium**:
   - 4 decoder layers (vs 24 in teacher model)
   - Distilled from the Medium Whisper ASR model

2. **Distilled Small**:
   - 4 decoder layers (vs 12 in teacher model)
   - Distilled from the Small Whisper ASR model

Both distilled models were trained on a combination of Common Voice v13, Gowajee, Thai Elderly Speech Corpus, custom scraped data, and Thai-Central Dialect from SLSCU Thai Dialect Corpus.

Use these models for efficient Thai ASR in resource-constrained environments or for faster inference times.

## Long-form Inference

Thonburian Whisper can be used for long-form audio transcription by combining VAD, Thai-word tokenizer, and chunking for word-level alignment.
We found that this is more robust and produce less insertion error rate (IER) comparing to using Whisper with timestamp. See `README.md` in [longform_transcription](https://github.com/biodatlab/thonburian-whisper/tree/main/longform_transcription) folder for detail usage.


## Developers

- [Biomedical and Data Lab, Mahidol University](https://biodatlab.github.io/)
- [WordSense](https://www.facebook.com/WordsenseAI) by [Looloo technology](https://loolootech.com/)

<p align="center">
  <img width="50px" src="assets/wordsense-looloo.png" />
  <img width="150px" src="assets/looloo-logo.png" />
</p>

## Citation

Our work was presented at the 7th ICNLSP 2024. You can find our paper [here](https://aclanthology.org/2024.icnlsp-1.17/). If you find our Thonburian Whisper models useful in your research, please consider citing:

```
@inproceedings{aung-etal-2024-thonburian,
    title = "Thonburian Whisper: Robust Fine-tuned and Distilled Whisper for {T}hai",
    author = "Aung, Zaw Htet and Thavornmongkol, Thanachot and Boribalburephan, Atirut and Tangsriworakan, Vittavas and Pipatsrisawat, Knot and Achakulvisut, Titipat",
    booktitle = "Proceedings of the 7th International Conference on Natural Language and Speech Processing (ICNLSP 2024)",
    year = "2024",
    pages = "149--156",
    url = "https://aclanthology.org/2024.icnlsp-1.17"
}
```

Or using the following written reference:

```
Zaw Htet Aung, Thanachot Thavornmongkol, Atirut Boribalburephan, Vittavas Tangsriworakan, Knot Pipatsrisawat, and Titipat Achakulvisut. 2024. Thonburian Whisper: Robust Fine-tuned and Distilled Whisper for Thai. In Proceedings of the 7th International Conference on Natural Language and Speech Processing (ICNLSP 2024), pages 149–156, Trento. Association for Computational Linguistics.
```
