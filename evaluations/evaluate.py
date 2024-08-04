import pandas as pd
import torch
from tqdm.auto import tqdm

pd.set_option("display.max_colwidth", None)
tqdm.pandas()


import string

from datasets import Audio, load_dataset
from jiwer import process_words, wer_default
from pythainlp.tokenize import word_tokenize as tokenize
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import evaluate

sampling_rate = 16_000
device = "cuda"
metric = evaluate.load("wer")


def clean_text(sentence: str, remove_punctuation: bool = True):
    sentence = sentence.strip()
    # Remove zero-width and non-breaking space.
    sentence = sentence.replace("\u200b", " ")
    sentence = sentence.replace("\xa0", " ")
    # remove redundant punctuations
    sentence = sentence.replace("เเ", "แ")

    # วรรณยุกต์/สระ
    sentence = sentence.replace("ํา", "ำ")
    sentence = sentence.replace("่ำ", "่ำ")
    sentence = sentence.replace("ำ้", "้ำ")
    sentence = sentence.replace("ํ่า", "่ำ")

    # replace special underscore and dash.
    sentence = sentence.replace("▁", "_")
    sentence = sentence.replace("—", "-")
    sentence = sentence.replace("–", "-")
    sentence = sentence.replace("−", "-")

    # replace special characters.
    sentence = sentence.replace("’", "'")
    sentence = sentence.replace("‘", "'")
    sentence = sentence.replace("”", '"')
    sentence = sentence.replace("“", '"')

    if remove_punctuation:
        sentence = "".join(
            [character for character in sentence if character not in string.punctuation]
        )
    return " ".join(sentence.split()).strip()


def compute_metrics_thai_text(
    pred_texts: list[str], ref_texts: list[str]
) -> dict[str, float | int]:
    """
    Compute the WER, IER, SER, and DER between the predicted and reference texts.
    Parameters
    ==========
    pred_texts: list[str]
        The list of predicted texts.
    ref_texts: list[str]
        The list of reference or ground truth texts.

    Returns
    =======
    dict
        A dictionary containing the WER, IER, SER, and DER.
    """
    # normalize everything and re-compute the WER
    norm_pred_texts = [clean_text(pred).lower() for pred in pred_texts]
    norm_ref_texts = [clean_text(label).lower() for label in ref_texts]

    # Since the `process_words` tokenizes the words based on space, we need to tokenize the Thai text and join them with space
    norm_pred_texts = [
        " ".join(tokenize(pred, engine="deepcut")) for pred in norm_pred_texts
    ]
    norm_ref_texts = [
        " ".join(tokenize(label, engine="deepcut")) for label in norm_ref_texts
    ]

    # filtering step to only evaluate the samples that correspond to non-zero normalized references:
    norm_pred_texts = [
        norm_pred_texts[i]
        for i in range(len(norm_pred_texts))
        if len(norm_ref_texts[i]) > 0
    ]
    norm_ref_texts = [
        norm_ref_texts[i]
        for i in range(len(norm_ref_texts))
        if len(norm_ref_texts[i]) > 0
    ]

    wer_output = process_words(
        norm_ref_texts, norm_pred_texts, wer_default, wer_default
    )
    wer_norm = 100 * wer_output.wer
    ier_norm = (
        100 * wer_output.insertions / sum([len(ref) for ref in wer_output.references])
    )
    ser_norm = (
        100
        * wer_output.substitutions
        / sum([len(ref) for ref in wer_output.references])
    )
    der_norm = (
        100 * wer_output.deletions / sum([len(ref) for ref in wer_output.references])
    )

    return {"wer": wer_norm, "ier": ier_norm, "ser": ser_norm, "der": der_norm}


@torch.no_grad()
def transcribe(
    processor,
    model,
    audio_dataset,
    batch_size: int = 8,
    sampling_rate: int = 16_000,
    device: str = "cuda",
):
    model.eval()
    model.to(device)
    all_predictions = []
    for i in tqdm(range(0, len(audio_dataset), batch_size)):
        audio_batch = audio_dataset[i : i + batch_size]
        input_speech_array_list = [
            audio_dict["array"] for audio_dict in audio_batch["audio"]
        ]

        inputs = processor(
            input_speech_array_list,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            # padding=True,
        )

        predicted_ids = model.generate(
            inputs["input_features"].to(device).half(),
            language="th",
            return_timestamps=False,
        )

        predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        all_predictions += predictions
    return all_predictions


if __name__ == "__main__":

    # Model init
    model_path = "biodatlab/whisper-th-medium-combined"

    processor = WhisperProcessor.from_pretrained(
        model_path, language="thai", task="transcribe", fast_tokenizer=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, use_flash_attention_2=True
    )
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="th", task="transcribe"
    )

    # Datset init
    test_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "th", split="test")
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.rename_column("sentence", "text")

    # Transcribe
    transcriptions = transcribe(
        processor,
        model,
        test_dataset,
        batch_size=32,
        device="cuda",
    )

    # Evaluate
    audio_transcript_df = pd.DataFrame(
        {"text": test_dataset["text"], "prediction": transcriptions}
    )
    audio_transcript_df.to_csv("whisper-medium-cmv13-test.csv")

    results = compute_metrics_thai_text(
        [p.lower() for p in audio_transcript_df["prediction"]],
        [t.lower() for t in test_dataset["text"]],
    )

    print(results)