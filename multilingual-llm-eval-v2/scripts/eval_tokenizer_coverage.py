import argparse
from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


palette = {
    "✅ Good": "green",
    "⚠️ Char-level fallback": "orange",
    "❌ Low coverage": "red"
}


def evaluate_tokenizer(tokenizer, tokenizer_name, dataset_name, lang_code, text_column, sample_size=1000):
    print(f"⏳ [{tokenizer_name}] Processing {lang_code}...")
    try:
        dataset = load_dataset(dataset_name, lang_code, split="train")
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    except Exception as e:
        print(f"⚠️  Skipping {lang_code} due to error: {e}")
        return None

    total = 0
    tokens = 0
    unk_tokens = 0
    total_token_chars = 0

    for example in dataset:
        text = example.get(text_column)
        if not isinstance(text, str):
            continue
        tokenized = tokenizer.tokenize(text)
        total += 1
        tokens += len(tokenized)
        unk_tokens += sum(1 for tok in tokenized if tok == tokenizer.unk_token)
        total_token_chars += sum(len(tok.replace("▁", "").replace("Ġ", "")) for tok in tokenized)

    if tokens == 0:
        return None

    coverage = 100 * (1 - unk_tokens / tokens)
    avg_token_len = total_token_chars / tokens

    return {
        "Language": lang_code,
        "Tokenizer": tokenizer_name,
        "Samples": total,
        "Total Tokens": tokens,
        "UNK Tokens": unk_tokens,
        "Token Coverage (%)": round(coverage, 2),
        "Avg Token Length": round(avg_token_len, 2)
    }


def plot_results(df, output_dir):
    plot_df = df.copy()
    plot_df.sort_values("Token Coverage (%)", ascending=True, inplace=True)

    plt.figure(figsize=(12, max(6, 0.35 * len(plot_df["Language"].unique()))))
    sns.barplot(
        data=plot_df,
        y="Language", x="Token Coverage (%)",
        hue="Performance Flag", palette=palette,
        orient="h"
    )
    plt.title("Tokenizer Coverage by Language (sorted with performance flags)")
    plt.xlabel("Token Coverage (%)")
    plt.ylabel("Language")
    plt.legend(title="Performance", loc="lower right")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "tokenizer_coverage_comparison_flagged.png")
    plt.savefig(plot_path)
    print(f"📊 Plot saved to: {plot_path}")


def plot_avg_token_length(df, output_dir):
    df_sorted = df.sort_values("Avg Token Length", ascending=True)
    plt.figure(figsize=(12, max(6, 0.35 * len(df_sorted["Language"].unique()))))
    sns.barplot(
        data=df_sorted,
        y="Language", x="Avg Token Length",
        hue="Performance Flag", palette=palette,
        orient="h"
    )
    plt.title("Average Token Length by Language (performance flagged)")
    plt.xlabel("Average Token Length")
    plt.ylabel("Language")
    plt.legend(title="Performance", loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "avg_token_length_comparison_flagged.png")
    plt.savefig(path)
    print(f"📏 Avg Token Length Plot saved to: {path}")

def flag_performance(row):
    if row["Token Coverage (%)"] > 90 and row["Avg Token Length"] < 2.0:
        return "⚠️ Char-level fallback"
    elif row["Token Coverage (%)"] < 50:
        return "❌ Low coverage"
    else:
        return "✅ Good"


def main(tokenizer_names, dataset_name, text_column, sample_size, output_dir, langs):
    os.makedirs(output_dir, exist_ok=True)
    lang_codes = langs if langs else get_dataset_config_names(dataset_name)
    results = []

    for tokenizer_name in tokenizer_names:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        for lang in lang_codes:
            result = evaluate_tokenizer(tokenizer, tokenizer_name, dataset_name, lang, text_column, sample_size)
            if result:
                results.append(result)

    if not results:
        print("❌ No results computed.")
        return

    df = pd.DataFrame(results)
    df.sort_values(by=["Language", "Tokenizer"], inplace=True)

    df["Performance Flag"] = df.apply(flag_performance, axis=1)



    # Save CSV
    csv_path = os.path.join(output_dir, "tokenizer_coverage_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved to: {csv_path}")

    palette = {
    "✅ Good": "green",
    "⚠️ Char-level fallback": "orange",
    "❌ Low coverage": "red"
}
    
    # Plot
    plot_results(df, output_dir)
    plot_avg_token_length(df, output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, nargs="+", required=True,
                        help="Tokenizer names (e.g. 'meta-llama/Llama-2-7b-hf mistralai/Mistral-7B-v0.1')")
    parser.add_argument("--dataset", type=str, default="Helsinki-NLP/opus-100", help="HuggingFace dataset ID")
    parser.add_argument("--text_column", type=str, default="text", help="Column containing the text")
    parser.add_argument("--samples", type=int, default=500, help="Max number of samples per language")
    parser.add_argument("--lang", nargs="+", default=None, help="Language codes to evaluate (e.g. am si lo)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save CSV and plots")
    args = parser.parse_args()

    main(
        tokenizer_names=args.tokenizer,
        dataset_name=args.dataset,
        text_column=args.text_column,
        sample_size=args.samples,
        output_dir=args.output_dir,
        langs=args.lang
    )

