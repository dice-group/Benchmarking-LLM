# General OPUS Evaluation Script (Translation)
# Supports: mBART, NLLB, mT5, LLaMA, BLOOM, Mistral, GPT
# Computes BLEU & chrF++, saves CSV, shows examples, plots histograms

import argparse
import csv
import random
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from metrics import compute_bleu, compute_chrfpp  # Ensure these are implemented

# List of decoder-only LLMs
LLM_MODELS = ["llama", "mistral", "bloom", "gpt", "gemma", "phi", "xglm"]
SEQ2SEQ_MODELS = ["mbart", "nllb", "mt5"]

# MBART/NLLB language codes
MBART_LANG_CODES = {
    "en": "en_XX", "fr": "fr_XX", "de": "de_DE", "ps": "ps_AF",
    "am": "am_ET", "ne": "ne_NP", "pa": "pa_IN", "sw": "sw_KE",
    "mr": "mr_IN", "yo": "yo_Latn", "ky": "kir_Cyrl", "kn": "kan_Deva",
    "tg": "tgk_Cyrl", "so": "som_Latn", "my": "my_MM", "si": "si_LK",
    "te": "te_IN"
}

def is_llm(model_name):
    return any(key in model_name.lower() for key in LLM_MODELS)

def is_seq2seq(model_name):
    return any(key in model_name.lower() for key in SEQ2SEQ_MODELS)

def prompt_translate(text, src_lang="English", tgt_lang="Swahili"):
    return (
        f"You are a professional translator.\n"
        f"Translate the following {src_lang} sentence into {tgt_lang}.\n\n"
        f"Input: {text}\nOutput:"
    )

def evaluate_opus(model_name, dataset_name, src_lang, tgt_lang, split, max_samples,
                  show_examples=5, output_csv="translation_samples.csv"):

    print(f"🔍 Loading dataset: {dataset_name}, {src_lang}->{tgt_lang}, split: {split}")
    try:
        dataset = load_dataset(dataset_name, f"{src_lang}-{tgt_lang}", split=f"{split}[:{max_samples}]")
    except Exception:
        print(f"⚠️ Could not load split '{split}', trying 'train' instead.")
        dataset = load_dataset(dataset_name, f"{src_lang}-{tgt_lang}", split=f"train[:{max_samples}]")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    if is_llm(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
    elif is_seq2seq(model_name):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer)

    predictions, references, sources = [], [], []
    bleu_scores, chrfpp_scores = [], []

    print("⚙️ Generating translations...")
    for example in dataset:
        if "translation" not in example:
            continue
        src_text = example["translation"].get(src_lang)
        ref_text = example["translation"].get(tgt_lang)
        if not src_text or not ref_text:
            continue

        try:
            if is_llm(model_name):
                prompt = prompt_translate(src_text, src_lang, tgt_lang)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=4,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    temperature=0.3,
                )
                pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Output:" in pred:
                    pred = pred.split("Output:")[-1].strip()

            elif is_seq2seq(model_name):
                # Set forced_bos_token_id if tokenizer has lang_code_to_id
                forced_bos = None
                if ("mbart" in model_name.lower() or "nllb" in model_name.lower()) and hasattr(tokenizer, "lang_code_to_id"):
                    src_code = MBART_LANG_CODES.get(src_lang, src_lang)
                    tgt_code = MBART_LANG_CODES.get(tgt_lang, tgt_lang)
                    tokenizer.src_lang = src_code
                    forced_bos = tokenizer.lang_code_to_id.get(tgt_code, None)

                inputs = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=512).to(device)
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=128,
                    num_beams=5,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                )
                pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

            else:
                pred = translator(src_text)[0]["translation_text"]

        except Exception as e:
            print(f"❌ Error translating sample: {e}")
            pred = ""

        predictions.append(pred)
        references.append(ref_text)
        sources.append(src_text)
        bleu_scores.append(compute_bleu([pred], [ref_text]))
        chrfpp_scores.append(compute_chrfpp([pred], [ref_text]))

    # --- Save translations to CSV ---
    print(f"\n💾 Saving {len(predictions)} translations to {output_csv}")
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Reference", "Prediction", "BLEU", "chrF++"])
        for src, ref, pred, bleu, chrf in zip(sources, references, predictions, bleu_scores, chrfpp_scores):
            writer.writerow([src, ref, pred, f"{bleu:.2f}", f"{chrf:.2f}"])

    # --- Show sample translations ---
    print("\n📊 Sample Translations:")
    for i in random.sample(range(len(predictions)), min(show_examples, len(predictions))):
        src, ref, pred = sources[i], references[i], predictions[i]
        ex_bleu, ex_chrfpp = bleu_scores[i], chrfpp_scores[i]
        print("\n────────────────────────────")
        print(f"Source:      {src}")
        print(f"Reference:   {ref}")
        print(f"Prediction:  {pred}")
        print(f"   → BLEU: {ex_bleu:.2f}, chrF++: {ex_chrfpp:.2f}")

    # --- Aggregate metrics ---
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_chrfpp = sum(chrfpp_scores) / len(chrfpp_scores)
    print("\n✅ Overall Evaluation Results:")
    print(f"   Average BLEU:   {avg_bleu:.2f}")
    print(f"   Average chrF++: {avg_chrfpp:.2f}")

    # --- Plot score distributions ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(bleu_scores, bins=20, color="#1f77b4", edgecolor="black")
    plt.title("Per-sentence BLEU Distribution")
    plt.xlabel("BLEU score")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(chrfpp_scores, bins=20, color="#ff7f0e", edgecolor="black")
    plt.title("Per-sentence chrF++ Distribution")
    plt.xlabel("chrF++ score")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="opus100")
    parser.add_argument("--source_lang", type=str, required=True)
    parser.add_argument("--target_lang", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--show_examples", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default="results/opus100/opus100_en-yo_mt5_eval.csv")
    args = parser.parse_args()

    evaluate_opus(
        model_name=args.model,
        dataset_name=args.dataset,
        src_lang=args.source_lang,
        tgt_lang=args.target_lang,
        split=args.split,
        max_samples=args.max_samples,
        show_examples=args.show_examples,
        output_csv=args.output_csv,
    )
