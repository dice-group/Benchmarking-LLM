# XL-Sum Evaluation Script (Summarization)
import argparse
from datasets import load_dataset
from transformers import pipeline
from metrics import compute_rouge

def evaluate_xlsum(model_name, lang, split, max_samples):
    print(f"🔍 Loading xlsum dataset for language '{lang}', split '{split}', max samples {max_samples}")
    try:
        dataset = load_dataset("csebuetnlp/xlsum", lang, split=f"{split}[:{max_samples}]")
    except Exception as e:
        print(f"⚠️ Could not load split '{split}', falling back to 'test'")
        dataset = load_dataset("csebuetnlp/xlsum", lang, split=f"test[:{max_samples}]")

    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name, device=1)

    predictions = []
    references = []

    for example in dataset:
        pred = summarizer(example["text"], max_length=128, truncation=True)[0]["summary_text"]
        predictions.append(pred)
        references.append(example["summary"])

    scores = compute_rouge(predictions, references)

    print(f"\n✅ ROUGE scores for {lang} ({split}):")
    for k, v in scores.items():
        print(f"  {k}: {v:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. facebook/mbart-large-50-many-to-many-mmt)")
    parser.add_argument("--lang", type=str, required=True, help="Language code for XLSum dataset (e.g. amharic, bengali)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (default: test)")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples to evaluate")
    args = parser.parse_args()

    evaluate_xlsum(args.model, args.lang, args.split, args.max_samples)
