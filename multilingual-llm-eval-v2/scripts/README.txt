

## Tokenizer Evaluation

This script evaluates the token coverage of a tokenizer across multiple languages.

**Usage:**

```bash
python scripts/eval_tokenizer.py \
  --tokenizer meta-llama/Llama-2-7b-hf \
  --dataset uonlp/CulturaX \
  --text_column text \
  --samples 1000 \
  --output tokenizer_coverage.csv




### ✅ Output

After running, you’ll get a file like `tokenizer_coverage.csv` with columns:

- `Language`
- `Tokenizer`
- `Samples`
- `Total Tokens`
- `UNK Tokens`
- `Token Coverage (%): 100%: Does not mean good tokenization`
- `Avg Token Length: ≈ 1.0–1.5: Suggests suboptimal handling of the script (i.e., character-level fallback)`
- `Performance Flag`

