# General OPUS Evaluation Script (Translation)
# Supports: mBART, NLLB, mT5, LLaMA, BLOOM, Mistral, GPT, Tiny Aya (CohereLabs/tiny-aya-*)
# Computes BLEU & chrF++, saves CSV, shows examples, plots histograms

import argparse
import csv
import os
import random
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from metrics import compute_bleu, compute_chrfpp  # Ensure these are implemented

# Decoder-only LLMs ("aya" matches CohereLabs/tiny-aya-global, -earth, -fire, -water)
LLM_MODELS = ["llama", "mistral", "bloom", "gpt", "gemma", "phi", "xglm", "aya"]
SEQ2SEQ_MODELS = ["mbart", "nllb", "mt5"]

# Instruction-tuned models that must be prompted through their chat template.
# Add other chat/instruct models here (e.g. "instruct", "-it") if you want the same treatment.
CHAT_TEMPLATE_MODELS = ["aya"]

# If True, LLM prompts use full language names ("Yoruba") instead of ISO codes ("yo").
# NOTE: this also changes the prompts of the other LLMs vs. your earlier runs.
USE_LANG_NAMES = True

# mBART-25 language codes (mBART only — do NOT reuse for NLLB, different scheme)
MBART_LANG_CODES = {
    "en": "en_XX", "fr": "fr_XX", "de": "de_DE", "ps": "ps_AF",
    "am": "am_ET", "ne": "ne_NP", "pa": "pa_IN", "sw": "sw_KE",
    "mr": "mr_IN", "yo": "yo_Latn", "ky": "kir_Cyrl", "kn": "kan_Deva",
    "tg": "tgk_Cyrl", "so": "som_Latn", "my": "my_MM", "si": "si_LK",
    "te": "te_IN"
}

# NLLB / FLORES-200 language codes (NLLB only — 3-letter ISO + script suffix)
NLLB_LANG_CODES = {
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "ps": "pbt_Arab",
    "am": "amh_Ethi", "ne": "npi_Deva", "pa": "pan_Guru", "sw": "swh_Latn",
    "mr": "mar_Deva", "yo": "yor_Latn", "ky": "kir_Cyrl", "kn": "kan_Knda",
    "tg": "tgk_Cyrl", "so": "som_Latn", "my": "mya_Mymr", "si": "sin_Sinh",
    "te": "tel_Telu",
}

# ISO code -> language name, used in LLM prompts
LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish", "ar": "Arabic",
    "hi": "Hindi", "af": "Afrikaans", "ps": "Pashto", "am": "Amharic", "ne": "Nepali",
    "pa": "Punjabi", "sw": "Swahili", "mr": "Marathi", "yo": "Yoruba", "ky": "Kyrgyz",
    "kn": "Kannada", "tg": "Tajik", "so": "Somali", "my": "Burmese", "si": "Sinhala",
    "te": "Telugu", "ha": "Hausa", "ig": "Igbo", "zu": "Zulu", "xh": "Xhosa",
    "rw": "Kinyarwanda",
}


def lang_name(code):
    return LANG_NAMES.get(code, code) if USE_LANG_NAMES else code


def is_llm(model_name):
    return any(key in model_name.lower() for key in LLM_MODELS)


def is_seq2seq(model_name):
    return any(key in model_name.lower() for key in SEQ2SEQ_MODELS)


def is_chat_model(model_name):
    return any(key in model_name.lower() for key in CHAT_TEMPLATE_MODELS)


def is_nllb(model_name):
    return "nllb" in model_name.lower()


def is_mbart(model_name):
    return "mbart" in model_name.lower()


def pick_dtype(model_name):
    """float32 on CPU; bfloat16 for chat models when the GPU supports it
    (fp16 can be unstable for newer LLM architectures); float16 otherwise."""
    if not torch.cuda.is_available():
        return torch.float32
    if is_chat_model(model_name) and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def prompt_translate(text, src_lang="English", tgt_lang="Swahili"):
    """Raw completion-style prompt for base / non-chat LLMs."""
    return (
        f"You are a professional translator.\n"
        f"Translate the following {src_lang} sentence into {tgt_lang}.\n\n"
        f"Input: {text}\nOutput:"
    )


def chat_translate_prompt(text, src_lang, tgt_lang):
    """User message for instruction-tuned chat models."""
    return (
        f"Translate the following {src_lang} text into {tgt_lang}. "
        f"Output only the {tgt_lang} translation, with no explanation.\n\n{text}"
    )


def translate_with_chat_model(model, tokenizer, device, text, src_lang, tgt_lang):
    """Translate with an instruction-tuned model (e.g. Tiny Aya) via its chat template."""
    messages = [{"role": "user", "content": chat_translate_prompt(text, lang_name(src_lang), lang_name(tgt_lang))}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    # The tokenizer returns token_type_ids, which decoder-only models like Tiny Aya
    # don't accept; generate() raises an error if they are passed.
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,          # low-resource scripts tokenize into many tokens
            do_sample=False,
            num_beams=4,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Keep only the newly generated tokens (drop the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def get_forced_bos_token_id(model_name, tokenizer, src_lang, tgt_lang):
    """Resolve the forced BOS token id for mBART/NLLB, using the right
    language-code table for each, and without relying on the now-removed
    `tokenizer.lang_code_to_id` attribute on newer NLLB tokenizers."""
    if is_nllb(model_name):
        src_code = NLLB_LANG_CODES.get(src_lang)
        tgt_code = NLLB_LANG_CODES.get(tgt_lang)
        if src_code is None or tgt_code is None:
            print(f"⚠️ Missing NLLB code for '{src_lang}' or '{tgt_lang}' — check NLLB_LANG_CODES.")
            return None
        tokenizer.src_lang = src_code
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
        # convert_tokens_to_ids silently returns unk_token_id if the code string
        # doesn't exist in the vocab — catch that instead of failing silently
        if forced_bos == tokenizer.unk_token_id:
            print(f"⚠️ '{tgt_code}' not found in tokenizer vocab — check the FLORES-200 code.")
            return None
        return forced_bos

    elif is_mbart(model_name):
        src_code = MBART_LANG_CODES.get(src_lang)
        tgt_code = MBART_LANG_CODES.get(tgt_lang)
        if src_code is None or tgt_code is None:
            print(f"⚠️ Missing mBART code for '{src_lang}' or '{tgt_lang}' — check MBART_LANG_CODES.")
            return None
        tokenizer.src_lang = src_code
        return tokenizer.lang_code_to_id.get(tgt_code)

    return None


def evaluate_opus(model_name, dataset_name, src_lang, tgt_lang, split, max_samples,
                  show_examples=5, output_csv="translation_samples.csv"):

    print(f"🔍 Loading dataset: {dataset_name}, {src_lang}->{tgt_lang}, split: {split}")
    try:
        dataset = load_dataset(dataset_name, f"{src_lang}-{tgt_lang}", split=f"{split}[:{max_samples}]")
    except Exception:
        print(f"⚠️ Could not load split '{split}', trying 'train' instead.")
        dataset = load_dataset(dataset_name, f"{src_lang}-{tgt_lang}", split=f"train[:{max_samples}]")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    translator = None
    # Load model
    if is_llm(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif is_seq2seq(model_name):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
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
            if is_llm(model_name) and is_chat_model(model_name):
                # Tiny Aya & other instruction-tuned chat models
                pred = translate_with_chat_model(model, tokenizer, device, src_text, src_lang, tgt_lang)

            elif is_llm(model_name):
                # Base / non-chat decoder-only LLMs (original behaviour)
                prompt = prompt_translate(src_text, lang_name(src_lang), lang_name(tgt_lang))
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                inputs.pop("token_type_ids", None)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=4,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    temperature=0.3,
                    pad_token_id=tokenizer.pad_token_id,
                )
                pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Output:" in pred:
                    pred = pred.split("Output:")[-1].strip()

            elif is_seq2seq(model_name):
                forced_bos = None
                if is_mbart(model_name) or is_nllb(model_name):
                    forced_bos = get_forced_bos_token_id(model_name, tokenizer, src_lang, tgt_lang)

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
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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
    parser.add_argument("--output_csv", type=str, default="results/opus100/opus100_en-yo_eval.csv")
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