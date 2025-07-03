# scripts/eval_belebele.py

from datasets import load_dataset
from transformers import pipeline
from scripts.metrics import compute_accuracy

def evaluate_belebele(model_name="facebook/bart-large-mnli", lang="eng_Latn", max_samples=50):
    # Load subset of the Belebele dataset
    dataset = load_dataset("facebook/belebele", lang, split=f"test[:{max_samples}]")
    qa_pipe = pipeline("text-classification", model=model_name, tokenizer=model_name, return_all_scores=True, device=0)

    predictions = []
    references = []

    for example in dataset:
        context = example["context"]
        question = example["question"]
        options = example["mc_options"]
        gold = example["correct_option"]

        scores = []
        for option in options:
            prompt = f"Q: {question}\nA: {option}\nContext: {context}"
            result = qa_pipe(prompt)
            scores.append(result[0][0]['score'])  # Take score of top label

        pred_index = int(scores.index(max(scores)))
        predictions.append(pred_index)
        references.append(gold)

    accuracy = compute_accuracy(predictions, references)
    print(f"✅ Belebele Evaluation for {lang}")
    print(f"   Samples: {len(dataset)}")
    print(f"   Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/bart-large-mnli")
    parser.add_argument("--lang", type=str, default="eng_Latn")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    evaluate_belebele(args.model, args.lang, args.max_samples)
