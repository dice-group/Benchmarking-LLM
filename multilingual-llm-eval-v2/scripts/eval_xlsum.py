# Summarization Evaluation Script
# Supports encoder-decoder models and decoder-only LLMs (e.g., LLaMA, Mistral)

import argparse
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from metrics import compute_rouge, compute_bleu  # assume you have these implemented
import torch
import os

LLM_MODELS = ["llama", "mistral", "bloom", "gpt", "gemma", "phi"]

def is_llm(model_name):
    return any(key in model_name.lower() for key in LLM_MODELS)

def prompt_summarize(text, language="English"):
    return f"Summarize the following {language} text:\n{text}\nSummary:"

def evaluate_summarization(model_path, dataset_name, text_field, summary_field, split, max_samples, language):
    print(f"🔍 Loading dataset: {dataset_name}, language: {language}, split: {split}")
    dataset = load_dataset(dataset_name, language.lower(), split=f"{split}[:{max_samples}]")

    print(f"📦 Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Choose model class based on type
    if is_llm(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
           # local_files_only=True,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            #local_files_only=True,
        )
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    predictions, references = [], []

    for example in dataset:
        text = example.get(text_field)
        ref_summary = example.get(summary_field)

        if not text or not ref_summary:
            continue

        if is_llm(model_path):
            prompt = prompt_summarize(text, language=language)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=256)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            try:
                result = summarizer(text, max_length=128, min_length=30, do_sample=False)
                pred = result[0]["summary_text"]
            except Exception as e:
                print(f"Error during summarization: {e}")
                continue

        predictions.append(pred)
        references.append(ref_summary)

    print(f"\n✅ Evaluation Results:")
    print(f"   ROUGE: {compute_rouge(predictions, references)}")
    print(f"   BLEU: {compute_bleu(predictions, references):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to local or remote summarization model")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--text_field", type=str, default="document", help="Field name of the text to summarize")
    parser.add_argument("--summary_field", type=str, default="summary", help="Field name of the reference summary")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (default: test)")
    parser.add_argument("--max_samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--language", type=str, default="English", help="Language of the input text")

    args = parser.parse_args()

    evaluate_summarization(
        model_path=args.model,
        dataset_name=args.dataset,
        text_field=args.text_field,
        summary_field=args.summary_field,
        split=args.split,
        max_samples=args.max_samples,
        language=args.language,
    )

