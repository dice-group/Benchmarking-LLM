# ✅ Summarization Evaluation Script for XL-Sum
# Supports encoder-decoder models and decoder-only LLMs (e.g., LLaMA, Mistral, Bloom, Qwen, XGLM)

import argparse
import csv
import torch
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from metrics import compute_rouge, compute_bleu  # your existing metrics

LLM_MODELS = ["llama", "mistral", "bloom", "gpt", "gemma", "phi", "qwen", "xglm"]

def is_llm(model_name):
    return any(key in model_name.lower() for key in LLM_MODELS)

def prompt_summarize(text, language="English"):
    return f"Summarize the following {language} text:\n{text}\n\nSummary:"

def evaluate_summarization(model_path, split="test", max_samples=100, language="English", output_csv="summaries.csv"):
    print(f"🔍 Loading XL-Sum dataset ({language}), split: {split}")
    dataset = load_dataset("csebuetnlp/xlsum", language, split=f"{split}[:{max_samples}]")

    print(f"📦 Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if is_llm(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        # ✅ Let pipeline manage device automatically (no `device=` here)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    predictions, references, sources = [], [], []

    print("⚙️ Generating summaries...")
    for i, example in enumerate(dataset):
        text = example["text"]
        ref_summary = example["summary"]

        try:
            if is_llm(model_path):
                # LLM-based summarization via prompt
                prompt = prompt_summarize(text, language)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                # ⚠️ Don't move to model.device (model is sharded)
                outputs = model.generate(**inputs, max_new_tokens=256)
                pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            else:
                # Seq2Seq summarization
                result = summarizer(text, max_length=128, min_length=30, do_sample=False)
                pred = result[0]["summary_text"]

        except Exception as e:
            print(f"❌ Error summarizing sample {i}: {e}")
            pred = ""

        predictions.append(pred)
        references.append(ref_summary)
        sources.append(text)

    # 💾 Save results to CSV
    print(f"\n💾 Saving {len(predictions)} summaries to {output_csv}")
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Source Text", "Reference Summary", "Generated Summary"])
        for src, ref, pred in zip(sources, references, predictions):
            writer.writerow([src, ref, pred])

    # 🧮 Compute metrics
    print(f"\n✅ Evaluation Results on {len(predictions)} samples:")
    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    print(f"   ROUGE-1: {rouge['ROUGE-1']:.2f}")
    print(f"   ROUGE-L: {rouge['ROUGE-L']:.2f}")
    print(f"   BLEU: {bleu:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or name of summarization model")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--language", type=str, default="english", help="Language subset from XL-Sum")
    parser.add_argument("--output_csv", type=str, default="summaries.csv", help="Where to save CSV results")
    args = parser.parse_args()

    evaluate_summarization(
        model_path=args.model,
        split=args.split,
        max_samples=args.max_samples,
        language=args.language,
        output_csv=args.output_csv,
    )
