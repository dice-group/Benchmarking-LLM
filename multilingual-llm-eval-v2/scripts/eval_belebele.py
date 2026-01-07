import torch
import csv
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from metrics import (
    compute_accuracy,
    compute_em,
    compute_f1,
    compute_bleu,
    compute_chrfpp,
    compute_rouge,
)
import math


# --- Model type checks ---
def is_xglm(model_name):
    return "xglm" in model_name.lower()


def is_llm(model_name):
    return any(x in model_name.lower() for x in ["llama", "mistral", "bloom", "qwen", "phi", "gemma"])


def is_seq2seq(model_name):
    return any(x in model_name.lower() for x in ["t5", "mt5", "bart", "mbart"])


# --- XGLM scoring ---
def score_with_xglm(prompt, answer, model, tokenizer):
    full_input = prompt + " " + answer
    inputs = tokenizer(full_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_likelihood = -outputs.loss.item() * inputs["input_ids"].shape[1]
    return log_likelihood


# --- General LLM scoring ---
def score_with_llm(prompt, answer, model, tokenizer):
    full_input = prompt + " " + answer
    inputs = tokenizer(full_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_likelihood = -outputs.loss.item() * inputs["input_ids"].shape[1]
    return log_likelihood


# --- Seq2Seq scoring ---
def score_with_seq2seq(prompt, answer, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    labels = tokenizer(answer, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        log_likelihood = -outputs.loss.item() * labels.shape[1]
    return log_likelihood


# --- Evaluation function ---
def evaluate_belebele(
    model_name="facebook/bart-large-mnli",
    lang="eng_Latn",
    max_samples=100,
    output_csv="belebele_results.csv",
):
    dataset = load_dataset("facebook/belebele", lang, split=f"test[:{max_samples}]")

    # --- Load model ---
    print(f"📦 Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_xglm(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        scorer = lambda p, a: score_with_xglm(p, a, model, tokenizer)
        model_type = "XGLM"

    elif is_llm(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        scorer = lambda p, a: score_with_llm(p, a, model, tokenizer)
        model_type = "LLM"

    elif is_seq2seq(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        scorer = lambda p, a: score_with_seq2seq(p, a, model, tokenizer)
        model_type = "Seq2Seq"

    else:
        qa_pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            device=0 if torch.cuda.is_available() else -1,
        )
        scorer = None
        model_type = "Classifier"

    print(f"✅ Model type detected: {model_type}")
    print(f"🌍 Language: {lang}")
    print(f"📊 Samples to evaluate: {len(dataset)}\n")

    predictions, references, contexts, questions, gold_options = [], [], [], [], []

    # --- Evaluation loop ---
    for example in dataset:
        context = example["flores_passage"]
        question = example["question"]
        options = [example["mc_answer1"], example["mc_answer2"],
                   example["mc_answer3"], example["mc_answer4"]]
        gold = int(example["correct_answer_num"]) - 1

        prompt = f"Context: {context}\nQ: {question}\nA:"

        scores = []
        for option in options:
            try:
                if scorer:
                    score = scorer(prompt, option)
                else:
                    result = qa_pipe(f"{prompt} {option}")
                    score = result[0]['score'] if isinstance(result[0], dict) else result[0][0]['score']
            except Exception as e:
                print(f"⚠️ Error scoring: {e}")
                score = -math.inf
            scores.append(score)

        pred_index = int(scores.index(max(scores)))
        predictions.append(pred_index)
        references.append(gold)
        contexts.append(context)
        questions.append(question)
        gold_options.append(options[gold])

    pred_texts = [dataset[i][f"mc_answer{p+1}"] for i, p in enumerate(predictions)]
    ref_texts = [dataset[i][f"mc_answer{r+1}"] for i, r in enumerate(references)]

    # --- Compute metrics ---
    acc = compute_accuracy(predictions, references)
    em = compute_em(pred_texts, ref_texts)
    f1 = compute_f1(pred_texts, ref_texts)
    bleu = compute_bleu(pred_texts, ref_texts)
    chrf = compute_chrfpp(pred_texts, ref_texts)
    rouge = compute_rouge(pred_texts, ref_texts)

    # --- Print results ---
    print(f"\n✅ Belebele Evaluation Results ({lang}) using {model_name}")
    print(f"   Accuracy:  {acc:.2f}%")
    print(f"   EM Score:  {em:.2f}")
    print(f"   F1 Score:  {f1:.2f}")
    print(f"   BLEU:      {bleu:.2f}")
    print(f"   chrF++:    {chrf:.2f}")
    print(f"   ROUGE-1:   {rouge['ROUGE-1']:.2f}")
    print(f"   ROUGE-L:   {rouge['ROUGE-L']:.2f}")

    # --- Save samples ---
    print(f"\n💾 Saving {len(dataset)} evaluation samples to {output_csv}")
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Context", "Question", "Gold Answer", "Predicted Answer"])
        for ctx, q, gold_ans, pred_ans in zip(contexts, questions, gold_options, pred_texts):
            writer.writerow([ctx, q, gold_ans, pred_ans])


# --- Main ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/xglm-564M")
    parser.add_argument("--lang", type=str, default="eng_Latn")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_csv", type=str, default="belebele_results.csv")
    args = parser.parse_args()

    evaluate_belebele(
        model_name=args.model,
        lang=args.lang,
        max_samples=args.max_samples,
        output_csv=args.output_csv,
    )
