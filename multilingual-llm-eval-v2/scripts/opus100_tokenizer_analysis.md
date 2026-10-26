**Table 16.** Tokenizer analysis on opus text for the 11 evaluation languages, by tokenizer. We report four metrics: (a) average tokens per word, (b) average out-of-vocabulary (OOV) rate, (c) average vocabulary coverage, and (d) average subword fragmentation. Arrows indicate the better direction; per language, the best value across tokenizers is in **bold**. Languages are grouped by script (L–Latin, NL–non-Latin). All runs use each model's own pretrained tokenizer on the same texts (up to 10,000 analysis texts per language); tokenizer vocabulary sizes are NLLB 256K, mBART/mt5-base 250K, Llama-3 128K, Mistral 33K, BLOOM 251K, Gemma-3 262K, and Aya 255K.

**(a) Avg. tokens / word (↓)**

| Language | NLLB | mBART | Llama-3 | Mistral | BLOOM | Gemma-3 | Aya | mt5-base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Yoruba (L) | 2.67 | 3.66 | 3.90 | 4.54 | **2.40** | 3.49 | 3.75 | 3.21 |
| Pashto (NL) | 2.41 | **2.19** | 3.48 | 4.97 | 3.32 | 2.64 | 3.41 | 2.45 |
| Kannada (NL) | 2.74 | 2.73 | 11.83 | 8.91 | **2.39** | 3.14 | 11.28 | 2.79 |
| Sinhala (NL) | 2.05 | **1.86** | 9.91 | 8.70 | 9.34 | 3.00 | 8.97 | 2.36 |
| Marathi (NL) | 2.01 | 1.98 | 3.36 | 5.26 | **1.87** | 2.01 | 4.20 | 2.41 |
| Punjabi (NL) | 2.10 | 2.09 | 6.82 | 9.29 | **1.85** | 2.83 | 6.84 | 2.55 |
| Tajik (NL) | **1.94** | 2.55 | 3.17 | 3.20 | 3.59 | 2.53 | 2.93 | 2.14 |
| Kyrgyz (NL) | 2.55 | **2.33** | 3.94 | 4.02 | 4.24 | 2.87 | 3.29 | 2.46 |
| Telugu (NL) | 2.23 | 2.18 | 10.84 | 10.00 | **2.16** | 2.62 | 10.39 | 2.55 |
| Burmese (NL) | **3.97** | 4.00 | 21.76 | 17.17 | 19.46 | 5.13 | 17.04 | 4.06 |
| Nepali (NL) | 2.08 | 2.01 | 3.63 | 6.10 | **1.92** | 2.25 | 4.83 | 2.49 |

**(b) Avg. OOV rate (↓)**

| Language | NLLB | mBART | Llama-3 | Mistral | BLOOM | Gemma-3 | Aya | mt5-base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Yoruba (L) | 0.84 | 0.91 | 0.92 | 0.94 | **0.83** | 0.91 | 0.91 | 0.90 |
| Pashto (NL) | 0.87 | **0.84** | 0.89 | 0.92 | 0.88 | 0.88 | 0.89 | 0.88 |
| Kannada (NL) | 0.92 | 0.91 | 0.94 | 0.95 | **0.89** | 0.92 | 0.93 | 0.93 |
| Sinhala (NL) | 0.93 | **0.88** | 0.97 | 0.98 | 0.97 | 0.97 | 0.98 | 0.95 |
| Marathi (NL) | 0.85 | 0.84 | 0.93 | 0.95 | **0.81** | 0.84 | 0.93 | 0.92 |
| Punjabi (NL) | 0.83 | 0.83 | 0.86 | 0.90 | **0.78** | 0.85 | 0.86 | 0.86 |
| Tajik (NL) | **0.92** | 0.93 | 0.94 | 0.96 | 0.95 | 0.93 | 0.93 | 0.93 |
| Kyrgyz (NL) | 0.86 | **0.83** | 0.90 | 0.93 | 0.91 | 0.86 | 0.86 | 0.88 |
| Telugu (NL) | 0.91 | **0.89** | 0.95 | 0.97 | **0.89** | 0.93 | 0.96 | 0.94 |
| Burmese (NL) | 0.95 | **0.94** | 0.95 | 0.97 | 0.95 | 0.95 | 0.95 | **0.94** |
| Nepali (NL) | 0.86 | 0.85 | 0.93 | 0.95 | **0.83** | 0.87 | 0.93 | 0.92 |

**(c) Avg. vocab. coverage (↑)**

| Language | NLLB | mBART | Llama-3 | Mistral | BLOOM | Gemma-3 | Aya | mt5-base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Yoruba (L) | 0.16 | 0.09 | 0.08 | 0.06 | **0.17** | 0.09 | 0.09 | 0.10 |
| Pashto (NL) | 0.13 | **0.16** | 0.11 | 0.08 | 0.12 | 0.12 | 0.11 | 0.12 |
| Kannada (NL) | 0.08 | 0.09 | 0.06 | 0.05 | **0.11** | 0.08 | 0.07 | 0.07 |
| Sinhala (NL) | 0.07 | **0.12** | 0.03 | 0.02 | 0.03 | 0.03 | 0.02 | 0.05 |
| Marathi (NL) | 0.15 | 0.16 | 0.07 | 0.05 | **0.19** | 0.16 | 0.07 | 0.08 |
| Punjabi (NL) | 0.17 | 0.17 | 0.14 | 0.10 | **0.22** | 0.15 | 0.14 | 0.14 |
| Tajik (NL) | **0.08** | 0.07 | 0.06 | 0.04 | 0.05 | 0.07 | 0.07 | 0.07 |
| Kyrgyz (NL) | 0.14 | **0.17** | 0.10 | 0.07 | 0.09 | 0.14 | 0.14 | 0.12 |
| Telugu (NL) | 0.09 | **0.11** | 0.05 | 0.03 | **0.11** | 0.07 | 0.04 | 0.06 |
| Burmese (NL) | 0.05 | **0.06** | 0.05 | 0.03 | 0.05 | 0.05 | 0.05 | **0.06** |
| Nepali (NL) | 0.14 | 0.15 | 0.07 | 0.04 | **0.17** | 0.13 | 0.07 | 0.08 |

**(d) Avg. subword fragmentation (↓)**

| Language | NLLB | mBART | Llama-3 | Mistral | BLOOM | Gemma-3 | Aya | mt5-base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Yoruba (L) | 0.56 | 0.80 | 0.85 | 0.89 | **0.55** | 0.79 | 0.80 | 0.77 |
| Pashto (NL) | 0.67 | **0.59** | 0.80 | 0.93 | 0.78 | 0.73 | 0.80 | 0.74 |
| Kannada (NL) | 0.74 | 0.72 | 0.88 | 0.89 | **0.69** | 0.79 | 0.88 | 0.84 |
| Sinhala (NL) | 0.55 | **0.50** | 0.95 | 0.95 | 0.95 | 0.87 | 0.95 | 0.81 |
| Marathi (NL) | 0.57 | 0.57 | 0.89 | 0.92 | **0.55** | 0.61 | 0.89 | 0.80 |
| Punjabi (NL) | 0.56 | 0.55 | 0.84 | 0.85 | **0.50** | 0.69 | 0.84 | 0.77 |
| Tajik (NL) | **0.50** | 0.69 | 0.78 | 0.79 | 0.82 | 0.69 | 0.78 | 0.64 |
| Kyrgyz (NL) | 0.73 | **0.68** | 0.85 | 0.89 | 0.87 | 0.79 | 0.82 | 0.79 |
| Telugu (NL) | 0.58 | **0.56** | 0.93 | 0.94 | 0.58 | 0.72 | 0.93 | 0.84 |
| Burmese (NL) | 0.90 | 0.89 | 0.89 | 0.96 | **0.88** | 0.93 | 0.95 | 0.90 |
| Nepali (NL) | 0.55 | 0.52 | 0.89 | 0.92 | **0.51** | 0.63 | 0.89 | 0.81 |
