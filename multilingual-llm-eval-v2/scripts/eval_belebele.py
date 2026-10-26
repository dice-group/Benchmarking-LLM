import math
import csv
import traceback
import torch
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoConfig,
    AutoTokenizer,
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

LLM_NAME_HINTS = ["llama", "mistral", "bloom", "qwen", "phi", "gemma", "aya", "cohere", "falcon"]
SEQ2SEQ_NAME_HINTS = ["t5", "bart"]


# --- Model type detection ---------------------------------------------------
def detect_model_type(model_name):
    """Decide how to score a model from its declared architecture, not its name.
    The old name-based check sent unknown models (e.g. aya-expanse) to the
    text-classification pipeline, which crashes for causal LMs."""
    name = model_name.lower()
    if "xglm" in name:
        return "XGLM"

    archs, is_enc_dec = "", False
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        archs = " ".join(cfg.architectures or [])
        is_enc_dec = bool(getattr(cfg, "is_encoder_decoder", False))
    except Exception as e:
        print(f"⚠️ Could not read config for {model_name} ({e}); falling back to name matching.")

    if "ForSequenceClassification" in archs:
        return "Classifier"          # checked first: bart-large-mnli is encoder-decoder too
    if is_enc_dec:
        return "Seq2Seq"             # T5, mT5, BART, mBART...
    if any(k in archs for k in ["ForCausalLM", "LMHeadModel", "ForConditionalGeneration"]):
        return "LLM"

    # Fallback when the config is unavailable
    if any(x in name for x in LLM_NAME_HINTS):
        return "LLM"
    if any(x in name for x in SEQ2SEQ_NAME_HINTS):
        return "Seq2Seq"
    return "Classifier"


# --- Scoring -----------------------------------------------------------------
@torch.no_grad()
def score_with_llm(prompt, answer, model, tokenizer, length_normalize=False):
    """Log-likelihood of the ANSWER tokens given the prompt (used for XGLM too).
    length_normalize=False -> sum of log-probs; True -> mean log-prob per token."""
    full_ids = tokenizer(prompt + " " + answer, return_tensors="pt").input_ids
    n_prompt = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    n_ans = full_ids.shape[1] - n_prompt
    if n_ans < 1:
        raise ValueError("Answer produced no extra tokens after the prompt.")

    full_ids = full_ids.to(model.device)
    # position j predicts token j+1, so the answer tokens are predicted by the
    # n_ans positions just before the last one
    logits = model(input_ids=full_ids).logits[0, -(n_ans + 1):-1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = full_ids[0, -n_ans:]
    token_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (token_lp.mean() if length_normalize else token_lp.sum()).item()


@torch.no_grad()
def score_with_seq2seq(prompt, answer, model, tokenizer, length_normalize=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    labels = tokenizer(answer, return_tensors="pt").input_ids.to(model.device)
    loss = model(**inputs, labels=labels).loss      # mean NLL over label tokens
    return (-loss if length_normalize else -loss * labels.shape[1]).item()


# --- Evaluation function -----------------------------------------------------
def evaluate_belebele(
    model_name="facebook/xglm-564M",
    lang="eng_Latn",
    max_samples=100,
    output_csv="belebele_results.csv",
    length_normalize=False,
):
    dataset = load_dataset("facebook/belebele", lang, split=f"test[:{max_samples}]")

    print(f"📦 Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = detect_model_type(model_name)
    scorer, qa_pipe = None, None

    if model_type in ("XGLM", "LLM"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()
        scorer = lambda p, a: score_with_llm(p, a, model, tokenizer, length_normalize)

    elif model_type == "Seq2Seq":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        model.eval()
        scorer = lambda p, a: score_with_seq2seq(p, a, model, tokenizer, length_normalize)

    else:
        # Only meaningful for NLI-style sequence-classification models
        qa_pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            device=0 if torch.cuda.is_available() else -1,
        )

    print(f"✅ Model type detected: {model_type}")
    print(f"🌍 Language: {lang}")
    print(f"📊 Samples to evaluate: {len(dataset)}\n")

    predictions, references, contexts, questions, gold_options = [], [], [], [], []

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
                    score = result[0]["score"] if isinstance(result[0], dict) else result[0][0]["score"]
            except Exception:
                # Do NOT fall back to -inf: if every option fails, argmax silently picks
                # option 0 and the accuracy becomes meaningless. Stop and show the real error.
                traceback.print_exc()
                raise
            scores.append(score)

        predictions.append(int(max(range(len(scores)), key=scores.__getitem__)))
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

    print(f"\n✅ Belebele Evaluation Results ({lang}) using {model_name}")
    print(f"   Accuracy:  {acc:.2f}%")
    print(f"   EM Score:  {em:.2f}")
    print(f"   F1 Score:  {f1:.2f}")
    print(f"   BLEU:      {bleu:.2f}")
    print(f"   chrF++:    {chrf:.2f}")
    print(f"   ROUGE-1:   {rouge['ROUGE-1']:.2f}")
    print(f"   ROUGE-L:   {rouge['ROUGE-L']:.2f}")

    print(f"\n💾 Saving {len(dataset)} evaluation samples to {output_csv}")
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Context", "Question", "Gold Answer", "Predicted Answer"])
        for ctx, q, gold_ans, pred_ans in zip(contexts, questions, gold_options, pred_texts):
            writer.writerow([ctx, q, gold_ans, pred_ans])


# --- Main --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/xglm-564M")
    parser.add_argument("--lang", type=str, default="eng_Latn")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_csv", type=str, default="belebele_results.csv")
    parser.add_argument("--length_normalize", action="store_true",
                        help="Score options by mean per-token log-prob instead of the sum")
    args = parser.parse_args()

    evaluate_belebele(
        model_name=args.model,
        lang=args.lang,
        max_samples=args.max_samples,
        output_csv=args.output_csv,
        length_normalize=args.length_normalize,
    )