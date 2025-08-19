# OPUS Evaluation Script (Translation)
# Uses BLEU and chrF

import argparse
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from metrics import compute_bleu, compute_chrf
import torch


LLM_MODELS = ["llama", "mistral", "bloom", "gpt", "gemma", "phi"]


def is_llm(model_name):
    return any(key in model_name.lower() for key in LLM_MODELS)


def prompt_translate(text, source_lang="English", target_lang="French"):
    return f"Translate the following text from {source_lang} to {target_lang}:\n{text}\nTranslation:"

#login("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")
def evaluate_opus(model_name, dataset_name, src_lang, tgt_lang, split, max_samples):
    lang_pair = f"{src_lang}-{tgt_lang}"
    print(f"🔍 Loading dataset: {dataset_name}, language pair: {lang_pair}, split: {split}")

    try:
       # login("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")
        dataset = load_dataset(dataset_name, lang_pair, split=f"{split}[:{max_samples}]")
    except Exception as e:
        print(f"⚠️  Could not load split '{split}', trying 'train' instead.")
        dataset = load_dataset(dataset_name, lang_pair, split=f"train[:{max_samples}]")

    print(f"📦 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if is_llm(model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, device=0)

    predictions, references = [], []
    for example in dataset:
        if "translation" not in example:
            continue
        src_text = example["translation"].get(src_lang)
        ref_text = example["translation"].get(tgt_lang)
        if not src_text or not ref_text:
            continue

        if is_llm(model_name):
            prompt = prompt_translate(src_text, source_lang=src_lang, target_lang=tgt_lang)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=128)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            pred = translator(src_text)[0]["translation_text"]

        predictions.append(pred)
        references.append(ref_text)

    print(f"\n✅ Evaluation Results ({src_lang} → {tgt_lang}):")
    print(f"   BLEU: {compute_bleu(predictions, references):.2f}")
    print(f"   chrF: {compute_chrf(predictions, references):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Translation or LLM model name")
    parser.add_argument("--dataset", type=str, default="opus100", help="Dataset name (e.g. opus100, opus_books)")
    parser.add_argument("--source_lang", type=str, required=True, help="Source language code (e.g. en)")
    parser.add_argument("--target_lang", type=str, required=True, help="Target language code (e.g. fr)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (default: test)")
    parser.add_argument("--max_samples", type=int, default=500, help="Number of samples to evaluate")

    args = parser.parse_args()

    evaluate_opus(
        model_name=args.model,
        dataset_name=args.dataset,
        src_lang=args.source_lang,
        tgt_lang=args.target_lang,
        split=args.split,
        max_samples=args.max_samples
    )

