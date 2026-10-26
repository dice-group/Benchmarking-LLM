# Summarization Evaluation Script for XL-Sum
# Supports encoder-decoder models and decoder-only LLMs
# (LLaMA, Mistral, Bloom, Qwen, XGLM, Gemma, and Aya: CohereLabs/aya-expanse-*, tiny-aya-*)

import argparse
import collections
import csv
import re
import traceback
import unicodedata
import torch
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from metrics import compute_rouge, compute_bleu, compute_chrfpp  # your existing metrics

# Name hints, only used as a fallback when the model config can't be read
LLM_MODELS = ["llama", "mistral", "bloom", "gpt", "gemma", "phi", "qwen", "xglm", "aya", "cohere"]

# Instruction-tuned models prompted through their chat template
CHAT_TEMPLATE_MODELS = ["aya", "-it", "instruct", "-chat"]


# --- Model helpers -------------------------------------------------------------
def is_seq2seq(model_path):
    """Decide from the model config (encoder-decoder or not), not from the name."""
    try:
        return bool(getattr(AutoConfig.from_pretrained(model_path), "is_encoder_decoder", False))
    except Exception as e:
        print(f"⚠️ Could not read config for {model_path} ({e}); falling back to name matching.")
        return not any(key in model_path.lower() for key in LLM_MODELS)


def is_chat_model(model_path):
    return any(key in model_path.lower() for key in CHAT_TEMPLATE_MODELS)


def display_name(language):
    return language.replace("_", " ").title()      # "chinese_simplified" -> "Chinese Simplified"


def pick_dtype(model_path):
    if not torch.cuda.is_available():
        return torch.float32
    if is_chat_model(model_path) and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def truncate_to_tokens(text, tokenizer, max_tokens):
    """Truncate the ARTICLE, not the whole prompt, so the instruction survives."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


# --- Prompts -------------------------------------------------------------------
def chat_summarize_prompt(text, language):
    """User message for instruction-tuned chat models. Explicitly asks for the summary
    in the article's language (otherwise many models answer in English)."""
    not_english = ", not in English," if language.lower() != "english" else ""
    return (
        f"Summarize the following {language} article in one or two sentences. "
        f"Write the summary in {language}{not_english} and output only the summary, "
        f"with no preamble.\n\n{text}"
    )


def build_chat_messages(text, language, shots):
    messages = []
    for shot_text, shot_summary in shots:           # few-shot examples as earlier turns
        messages.append({"role": "user", "content": chat_summarize_prompt(shot_text, language)})
        messages.append({"role": "assistant", "content": shot_summary})
    messages.append({"role": "user", "content": chat_summarize_prompt(text, language)})
    return messages


def raw_block(text, language, summary=None):
    block = f"{language} text:\n{text}\n\nSummary in {language}:"
    return block + (f" {summary}\n\n###\n\n" if summary is not None else "")


def build_raw_prompt(text, language, shots):
    """Completion-style prompt for base (non-chat) LLMs."""
    header = (f"Summarize each {language} text in one or two sentences. "
              f"Write the summary in {language}.\n\n")
    return header + "".join(raw_block(t, language, s) for t, s in shots) + raw_block(text, language)


def load_shots(language, n_shots, tokenizer, max_tokens=300):
    """Few-shot examples: short-ish articles from the TRAIN split (never from the test set)."""
    if n_shots <= 0:
        return []
    pool = load_dataset("csebuetnlp/xlsum", language, split="train[:200]")
    candidates = sorted((ex for ex in pool if len(ex["text"]) > 300), key=lambda ex: len(ex["text"]))
    return [(truncate_to_tokens(ex["text"], tokenizer, max_tokens), ex["summary"])
            for ex in candidates[:n_shots]]


# --- Metrics helpers -------------------------------------------------------------
class UnicodeTokenizer:
    """Script-agnostic tokenizer for ROUGE.
    rouge_score's default tokenizer replaces everything outside [a-z0-9] with spaces, so
    Amharic (Ge'ez), Arabic, Hindi... text can become EMPTY. This keeps letters (L),
    combining marks (M) and digits (N) from any script."""

    def tokenize(self, text):
        text = unicodedata.normalize("NFC", text.lower())
        tokens, cur = [], []
        for ch in text:
            if unicodedata.category(ch)[0] in ("L", "M", "N"):
                cur.append(ch)
            elif cur:
                tokens.append("".join(cur))
                cur = []
        if cur:
            tokens.append("".join(cur))
        return tokens


def compute_rouge_unicode(predictions, references):
    """Mean ROUGE-1 / ROUGE-L F1 on a 0-100 scale. Needs `pip install rouge-score`."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], tokenizer=UnicodeTokenizer())
    r1, rl = [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    n = max(len(r1), 1)
    return {"ROUGE-1": 100 * sum(r1) / n, "ROUGE-L": 100 * sum(rl) / n}


def dominant_script(text):
    """Most frequent Unicode script name among the letters (e.g. ETHIOPIC, LATIN, ARABIC)."""
    counts = collections.Counter()
    for ch in text:
        if ch.isalpha():
            try:
                counts[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                pass
    return counts.most_common(1)[0][0] if counts else None


# --- Generation ----------------------------------------------------------------
@torch.no_grad()
def summarize_with_llm(model, tokenizer, model_path, text, language, shots, max_input_tokens, max_new_tokens):
    text = truncate_to_tokens(text, tokenizer, max_input_tokens)
    chat = is_chat_model(model_path)

    if chat:
        inputs = tokenizer.apply_chat_template(
            build_chat_messages(text, language, shots),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = tokenizer(build_raw_prompt(text, language, shots), return_tensors="pt").to(model.device)

    inputs.pop("token_type_ids", None)              # decoder-only models reject it

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Keep ONLY the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not chat:
        pred = re.split(r"\n\s*\n|###", pred)[0].strip()   # base models keep going after the summary
    return pred


def evaluate_summarization(model_path, split="test", max_samples=100, language="english",
                           output_csv="summaries.csv", max_input_tokens=1024, max_new_tokens=256,
                           n_shots=0):
    print(f"🔍 Loading XL-Sum dataset ({language}), split: {split}")
    dataset = load_dataset("csebuetnlp/xlsum", language, split=f"{split}[:{max_samples}]")
    lang_name = display_name(language)

    print(f"📦 Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    seq2seq = is_seq2seq(model_path)
    summarizer, shots = None, []

    if not seq2seq:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=pick_dtype(model_path),
        )
        model.eval()
        # Gemma 3 uses a compilable KV cache, so generate() tries to torch.compile the forward
        # pass, which fails with accelerate's device_map="auto" hooks. Turn auto-compile off.
        model.generation_config.disable_compile = True
        # Greedy decoding: clear sampling params so generate() stops warning about them
        model.generation_config.top_k = None
        model.generation_config.top_p = None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        shots = load_shots(language, n_shots, tokenizer)
        print(f"✅ Model type: decoder-only LLM "
              f"({'chat template' if is_chat_model(model_path) else 'raw prompt'}, {len(shots)}-shot)")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        print("✅ Model type: encoder-decoder")

    predictions, references, sources = [], [], []
    n_errors = 0

    print("⚙️ Generating summaries...")
    for i, example in enumerate(dataset):
        text = example["text"]
        ref_summary = example["summary"]

        try:
            if not seq2seq:
                pred = summarize_with_llm(model, tokenizer, model_path, text, lang_name, shots,
                                          max_input_tokens, max_new_tokens)
            else:
                result = summarizer(text, max_length=128, min_length=30, do_sample=False, truncation=True)
                pred = result[0]["summary_text"]

        except Exception as e:
            n_errors += 1
            print(f"❌ Error summarizing sample {i}: {e}")
            if n_errors == 1:
                traceback.print_exc()
            pred = ""

        predictions.append(pred)
        references.append(ref_summary)
        sources.append(text)

    if n_errors:
        print(f"\n⚠️ {n_errors}/{len(dataset)} samples failed and were scored as empty summaries; "
              f"the metrics below are lower than the model's true quality.")

    print(f"\n💾 Saving {len(predictions)} summaries to {output_csv}")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Source Text", "Reference Summary", "Generated Summary"])
        for src, ref, pred in zip(sources, references, predictions):
            writer.writerow([src, ref, pred])

    # Metrics
    print(f"\n✅ Evaluation Results on {len(predictions)} samples:")
    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    chrf = compute_chrfpp(predictions, references)
    print(f"   ROUGE-1 (your metrics.py):          {rouge['ROUGE-1']:.2f}")
    print(f"   ROUGE-L (your metrics.py):          {rouge['ROUGE-L']:.2f}")
    try:
        rouge_u = compute_rouge_unicode(predictions, references)
        print(f"   ROUGE-1 (unicode tokenizer, 0-100): {rouge_u['ROUGE-1']:.2f}")
        print(f"   ROUGE-L (unicode tokenizer, 0-100): {rouge_u['ROUGE-L']:.2f}")
    except ImportError:
        print("   (pip install rouge-score to also get script-agnostic ROUGE)")
    print(f"   BLEU:    {bleu:.2f}")
    print(f"   chrF++:  {chrf:.2f}")

    # Language/script sanity check: did the model answer in the reference's script?
    matches = sum(dominant_script(p) == dominant_script(r) and dominant_script(p) is not None
                  for p, r in zip(predictions, references))
    print(f"   Predictions in the same script as the reference: {100 * matches / max(len(predictions), 1):.0f}% "
          f"(only informative for non-Latin scripts)")

    print("\n🔎 First 3 generations:")
    for ref, pred in list(zip(references, predictions))[:3]:
        print(f"   REF : {ref[:200]}")
        print(f"   PRED: {pred[:200]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or name of summarization model")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--language", type=str, default="english", help="Language subset from XL-Sum (e.g. amharic)")
    parser.add_argument("--output_csv", type=str, default="summaries.csv", help="Where to save CSV results")
    parser.add_argument("--max_input_tokens", type=int, default=1024, help="Max article tokens given to LLMs")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generated tokens for LLMs")
    parser.add_argument("--n_shots", type=int, default=0, help="Few-shot examples from the train split (LLMs only)")
    args = parser.parse_args()

    evaluate_summarization(
        model_path=args.model,
        split=args.split,
        max_samples=args.max_samples,
        language=args.language,
        output_csv=args.output_csv,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        n_shots=args.n_shots,
    )