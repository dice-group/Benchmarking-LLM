# General OPUS Evaluation Script (Translation)
# Supports: mBART, NLLB, mT5, LLaMA, BLOOM, Mistral, GPT
# Computes BLEU & chrF++

import argparse
import torch
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

def evaluate_opus(model_name, dataset_name, src_lang, tgt_lang, split, max_samples):

    print(f"🔍 Loading dataset: {dataset_name}, {src_lang}->{tgt_lang}, split: {split}")
    try:
        dataset = load_dataset(dataset_name, f"{src_lang}-{tgt_lang}", split=f"{split}[:{max_samples}]")
    except Exception:
        print(f"⚠️ Could not load split '{split}', trying 'train' instead.")
        dataset = load_dataset(dataset_name, f"{src_lang}-{tgt_lang}", split=f"train[:{max_samples}]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    if is_llm(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    elif is_seq2seq(model_name):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer)

    predictions, references = [], []

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

    # --- Aggregate metrics ---
    avg_bleu = sum([compute_bleu([p], [r]) for p, r in zip(predictions, references)]) / len(predictions)
    avg_chrfpp = sum([compute_chrfpp([p], [r]) for p, r in zip(predictions, references)]) / len(predictions)

    print("\n✅ Overall Evaluation Results:")
    print(f"   Average BLEU:   {avg_bleu:.2f}")
    print(f"   Average chrF++: {avg_chrfpp:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="opus100")
    parser.add_argument("--source_lang", type=str, required=True)
    parser.add_argument("--target_lang", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    evaluate_opus(
        model_name=args.model,
        dataset_name=args.dataset,
        src_lang=args.source_lang,
        tgt_lang=args.target_lang,
        split=args.split,
        max_samples=args.max_samples,
    )
