import os
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from metrics import compute_em, compute_bleu, compute_chrf, compute_f1, compute_rouge
from loaders import load_dataset_local, load_culturax

def format_prompt(context, question, task):
    if task == "qa":
        return f"{context}\n\nQuestion: {question}\nAnswer:"
    elif task == "generation":
        return context
    elif task == "translation":
        return f"Translate this:\n{context}\n\nTranslation:"
    elif task == "summarization":
        return f"Summarize:\n{context}\n\nSummary:"
    else:
        raise ValueError("Unsupported task")

def evaluate(model_name, data_dir, lang, task, source):
    if source == "culturax":
        samples = load_culturax(lang, task)
    else:
        samples = load_dataset_local(data_dir, task, lang)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    predictions, references = [], []

    for context, question, reference in samples:
        prompt = format_prompt(context, question, task)
        output = pipe(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        result = output.split(":")[-1].strip()
        predictions.append(result)
        references.append(reference)

    print(f"\n[{lang.upper()} | {task.upper()} | {source}]")
    print(f"  EM Score:   {compute_em(predictions, references):.2f}%")
    print(f"  BLEU Score: {compute_bleu(predictions, references):.2f}")
    print(f"  chrF Score: {compute_chrf(predictions, references):.2f}")
    print(f"  F1 Score: {compute_f1(predictions, references):.2f}")
    rouge_scores = compute_rouge(predictions, references)
    print(f"  ROUGE-1: {rouge_scores['ROUGE-1']:.2f}")
    print(f"  ROUGE-L: {rouge_scores['ROUGE-L']:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=["qa", "generation", "translation", "summarization"])
    parser.add_argument("--lang", required=True)
    parser.add_argument("--data_dir", required=False, default="data")
    parser.add_argument("--source", choices=["local", "culturax"], default="local")
    args = parser.parse_args()

    evaluate(args.model, args.data_dir, args.lang, args.task, args.source)

if __name__ == "__main__":
    main()
