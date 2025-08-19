from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from metrics import (
    compute_accuracy,
    compute_em,
    compute_f1,
    compute_bleu,
    compute_chrfpp,
   # compute_rouge,
)
import torch
import math


def is_xglm(model_name):
    return "xglm" in model_name.lower()


def score_with_xglm(prompt, answer, model, tokenizer):
    full_input = prompt + " " + answer
    inputs = tokenizer(full_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_likelihood = -outputs.loss.item() * inputs["input_ids"].shape[1]
    return log_likelihood


def evaluate_belebele(model_name="facebook/bart-large-mnli", lang="eng_Latn", max_samples=50):
    dataset = load_dataset("facebook/belebele", lang, split=f"test[:{max_samples}]")
    use_xglm = is_xglm(model_name)

    if use_xglm:
        print(f"📦 Using XGLM model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print(f"📦 Using classifier model: {model_name}")
        qa_pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            device=0 if torch.cuda.is_available() else -1
        )

    predictions = []
    references = []

    for example in dataset:
        context = example["flores_passage"]
        question = example["question"]
        options = [example["mc_answer1"], example["mc_answer2"],
                   example["mc_answer3"], example["mc_answer4"]]
        gold = int(example["correct_answer_num"]) - 1

        scores = []
        for option in options:
            prompt = f"Context: {context}\nQ: {question}\nA:"
            if use_xglm:
                score = score_with_xglm(prompt, option, model, tokenizer)
            else:
                result = qa_pipe(f"{prompt} {option}")
                score = result[0]['score'] if isinstance(result[0], dict) else result[0][0]['score']
            scores.append(score)

        pred_index = int(scores.index(max(scores)))
        predictions.append(pred_index)
        references.append(gold)

    pred_texts = [dataset[i][f"mc_answer{p+1}"] for i, p in enumerate(predictions)]
    ref_texts = [dataset[i][f"mc_answer{r+1}"] for i, r in enumerate(references)]

    print(f"\n✅ Belebele Evaluation for {lang} using {model_name}")
    print(f"   Samples: {len(dataset)}")
    print(f"   Accuracy: {compute_accuracy(predictions, references):.2f}%")
    print(f"   EM Score: {compute_em(pred_texts, ref_texts):.2f}")
    print(f"   F1 Score: {compute_f1(pred_texts, ref_texts):.2f}")
    print(f"   BLEU Score: {compute_bleu(pred_texts, ref_texts):.2f}")
    print(f"   chrF++ Score: {compute_chrfpp(pred_texts, ref_texts):.2f}")
    #rouge = compute_rouge(pred_texts, ref_texts)
    #print(f"   ROUGE-1: {rouge['ROUGE-1']:.2f}")
   # print(f"   ROUGE-L: {rouge['ROUGE-L']:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/bart-large-mnli")
    parser.add_argument("--lang", type=str, default="eng_Latn")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    evaluate_belebele(args.model, args.lang, args.max_samples)

