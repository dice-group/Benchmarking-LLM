# Benchmarking-LLM
This is repository is about task-agnostic multilingual evaluation and benchmark flexibility over diverse script languages

## Multilingual LLM Evaluation

This script evaluates multilingual LLMs on low resources languages for different tasks using different datasets: CulturaX, Opus, XLSum and Belebele
### Tasks and Datasets
- `Text Generation (CulturaX)`
- `Machine Translation (Opus100)`
- `Text Summarization (XLSum)`
- `Question Answering (Belebele)`

### Selected Languages

<table>
  <tr><th align="left">Iso code</th><th>Language</th><th>Language script</th><th>Language class</th></tr>
  <tr><th align="left">am</th><td>Amharic</td><td>Ge'ez</td><td>2</td></tr>
  <tr><th align="left">te</th><td>Telugu</td><td>Devanagari</td><td>1</td></tr>
  <tr><th align="left">my</th><td>Burmese</td><td>Burmese</td><td>1</td></tr>
  <tr><th align="left">ne</th><td>Nepali</td><td>Devanagari</td><td> 1</td></tr>
  <tr><th align="left">kn</th><td>Kannada</td><td>Kannada</td><td>1</td></tr>
  <tr><th align="left">ps</th><td>Pashto</td><td>Arabic</td><td>1</td></tr>
  <tr><th align="left">tg</th><td>Tajik</td><td>Cyrillic</td><td>1</td></tr>
  tr><th align="left">sw</th><td>Swahili</td><td>Latin</td><td>2</td></tr>
  <tr><th align="left">yo</th><td>Yoruba</td><td>Latin</td><td>2</td></tr>
  <tr><th align="left">so</th><td>Somali</td><td>Latin</td><td>1</td></tr>
  <tr><th align="left">si</th><td>Sinhala</td><td>Sinhala</td><td>0</td></tr>
   <tr><th align="left">mr</th><td>Marathi</td><td>Devanagari</td><td>2</td></tr>
    <tr><th align="left">pa</th><td>Punjabi</td><td>Gurmukhi</td><td>2</td></tr>
     <tr><th align="left">ky</th><td>Kyrgyz</td><td>Cyrillic</td><td>2</td></tr>
</table>


### Selected Multilingual LLMs

<table>
  <tr><th align="left">Models</th><th>Tokenizer type</th><th>Task</th></tr>
  <tr><th align="left">LLama2</th><td>SentencePiece (BPE)</td><td>Generation, Translation , Summarization, QA</td></tr>
  <tr><th align="left">Mistral</th><td>SentencePiece (BPE)</td><td>Generation, Translation , Summarization, QA</td></tr>
  <tr><th align="left">XGLM</th><td>Byte-Pair Encoding (BPE)</td><td>Generation, Translation , Summarization, QA</td></tr>
  <tr><th align="left">BlOOM</th><td>Byte-level BPE</td><td>Generation, Translation , Summarization, QA</td></tr>
  <tr><th align="left">Qwen</th><td>tiktoken or SentencePiece</td><td>Generation, Translation , Summarization, QA</td></tr>
  <tr><th align="left">NLLB</th><td>SentencePiece (BPE)</td><td>Translation</td></tr>
  tr><th align="left">mBART-large</th><td>SentencePiece (BPE)</td><td>Translation</td></tr>
  <tr><th align="left">mT5-base</th><td>SentencePiece (Unigram)</td><td>Translation</td></tr>
</table>
**Usage:**

```bash
python scripts/eval_opus.py \
  --model meta-llama/Llama-2-7b-hf \
  --task generation \
  --lang am 
```

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
```



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
