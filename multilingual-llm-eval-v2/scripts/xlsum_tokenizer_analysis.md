**Table 16.** Tokenizer analysis on xlsum text for the 12 evaluation languages, by tokenizer. We report four metrics: (a) average tokens per word, (b) average out-of-vocabulary (OOV) rate, (c) average vocabulary coverage, and (d) average subword fragmentation. Arrows indicate the better direction; per language, the best value across tokenizers is in **bold**. Languages are grouped by script (L–Latin, NL–non-Latin). All runs use each model's own pretrained tokenizer on the same texts (up to 10,000 analysis texts per language); tokenizer vocabulary sizes are Llama-3 128K, Mistral 33K, BLOOM 251K, Gemma-3 262K, Aya 255K, and mt5-base/mt5-multi 250K. Somali, Swahili, Yoruba, Sinhala, Punjabi, Kyrgyz, Amharic, Burmese, and Nepali have fewer analysis texts available (5962, 7898, 6350, 3249, 8215, 2266, 5761, 4569, and 5808, respectively).

**(a) Avg. tokens / word (↓)**

| Language | Llama-3 | Mistral | BLOOM | Gemma-3 | Aya | mt5-base | mt5-multi |
|---|---:|---:|---:|---:|---:|---:|---:|
| Somali (L) | 2.46 | 2.76 | 2.34 | 2.20 | 2.32 | **1.90** | **1.90** |
| Swahili (L) | 2.41 | 2.71 | **1.51** | 2.06 | 2.25 | 1.78 | 1.78 |
| Yoruba (L) | 2.35 | 2.77 | **1.63** | 2.07 | 2.22 | 1.85 | 1.85 |
| Pashto (NL) | 2.72 | 4.01 | 2.57 | 1.90 | 2.64 | **1.81** | **1.81** |
| Sinhala (NL) | 11.65 | 10.02 | 10.99 | 3.09 | 10.56 | **2.48** | **2.48** |
| Marathi (NL) | 3.93 | 6.21 | **1.71** | 1.91 | 5.17 | 2.49 | 2.49 |
| Punjabi (NL) | 7.89 | 10.70 | **1.40** | 2.61 | 7.91 | 2.38 | 2.38 |
| Kyrgyz (NL) | 4.32 | 4.35 | 4.67 | 3.00 | 3.59 | **2.32** | **2.32** |
| Telugu (NL) | 13.85 | 12.35 | **2.17** | 2.87 | 13.24 | 2.73 | 2.73 |
| Amharic (NL) | 12.29 | 10.44 | 8.17 | 3.14 | 10.12 | **3.03** | **3.03** |
| Burmese (NL) | 22.37 | 17.65 | 20.24 | 4.94 | 17.77 | **3.33** | **3.33** |
| Nepali (NL) | 3.83 | 6.31 | **1.61** | 2.17 | 5.04 | 2.39 | 2.39 |

**(b) Avg. OOV rate (↓)**

| Language | Llama-3 | Mistral | BLOOM | Gemma-3 | Aya | mt5-base | mt5-multi |
|---|---:|---:|---:|---:|---:|---:|---:|
| Somali (L) | 0.97 | 0.98 | 0.96 | 0.96 | **0.95** | 0.96 | 0.96 |
| Swahili (L) | 0.96 | 0.98 | 0.95 | **0.94** | **0.94** | 0.95 | 0.95 |
| Yoruba (L) | 0.94 | 0.97 | **0.93** | **0.93** | **0.93** | **0.93** | **0.93** |
| Pashto (NL) | 0.99 | 1.00 | **0.97** | 0.98 | 0.98 | **0.97** | **0.97** |
| Sinhala (NL) | 0.99 | 0.99 | 0.98 | 0.99 | 0.99 | **0.97** | **0.97** |
| Marathi (NL) | 0.99 | 1.00 | **0.97** | 0.98 | 0.99 | 0.98 | 0.98 |
| Punjabi (NL) | 0.99 | 1.00 | **0.97** | 0.99 | 0.99 | 0.98 | 0.98 |
| Kyrgyz (NL) | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | **0.97** | **0.97** |
| Telugu (NL) | 1.00 | 1.00 | **0.99** | 1.00 | 1.00 | **0.99** | **0.99** |
| Amharic (NL) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.99** | **0.99** |
| Burmese (NL) | **0.99** | 1.00 | **0.99** | **0.99** | **0.99** | **0.99** | **0.99** |
| Nepali (NL) | 1.00 | 1.00 | **0.96** | 0.97 | 1.00 | 0.99 | 0.99 |

**(c) Avg. vocab. coverage (↑)**

| Language | Llama-3 | Mistral | BLOOM | Gemma-3 | Aya | mt5-base | mt5-multi |
|---|---:|---:|---:|---:|---:|---:|---:|
| Somali (L) | 0.03 | 0.02 | 0.04 | 0.04 | **0.05** | 0.04 | 0.04 |
| Swahili (L) | 0.04 | 0.02 | 0.05 | **0.06** | **0.06** | 0.05 | 0.05 |
| Yoruba (L) | 0.06 | 0.03 | **0.07** | **0.07** | **0.07** | **0.07** | **0.07** |
| Pashto (NL) | 0.01 | 0.004 | **0.03** | 0.02 | 0.02 | **0.03** | **0.03** |
| Sinhala (NL) | 0.01 | 0.007 | 0.02 | 0.01 | 0.01 | **0.03** | **0.03** |
| Marathi (NL) | 0.01 | 0.005 | **0.03** | 0.02 | 0.009 | 0.02 | 0.02 |
| Punjabi (NL) | 0.01 | 0.004 | **0.03** | 0.007 | 0.006 | 0.02 | 0.02 |
| Kyrgyz (NL) | 0.01 | 0.005 | 0.01 | 0.01 | 0.02 | **0.03** | **0.03** |
| Telugu (NL) | 0.003 | 0.001 | 0.008 | 0.003 | 0.002 | **0.009** | **0.009** |
| Amharic (NL) | 0.004 | 0.001 | 0.005 | 0.003 | 0.002 | **0.008** | **0.008** |
| Burmese (NL) | 0.007 | 0.004 | 0.007 | 0.008 | 0.008 | **0.009** | **0.009** |
| Nepali (NL) | 0.004 | 0.002 | **0.04** | 0.03 | 0.004 | 0.008 | 0.008 |

**(d) Avg. subword fragmentation (↓)**

| Language | Llama-3 | Mistral | BLOOM | Gemma-3 | Aya | mt5-base | mt5-multi |
|---|---:|---:|---:|---:|---:|---:|---:|
| Somali (L) | 0.71 | 0.83 | 0.66 | 0.61 | 0.68 | **0.57** | **0.57** |
| Swahili (L) | 0.75 | 0.79 | **0.33** | 0.60 | 0.71 | 0.57 | 0.57 |
| Yoruba (L) | 0.68 | 0.72 | **0.40** | 0.57 | 0.65 | 0.52 | 0.52 |
| Pashto (NL) | 0.74 | 0.91 | 0.69 | **0.54** | 0.71 | 0.58 | 0.58 |
| Sinhala (NL) | 0.99 | 1.00 | 0.99 | 0.82 | 1.00 | **0.81** | **0.81** |
| Marathi (NL) | 0.97 | 0.99 | **0.48** | 0.57 | 0.98 | 0.85 | 0.85 |
| Punjabi (NL) | 0.99 | 1.00 | **0.26** | 0.67 | 1.00 | 0.77 | 0.77 |
| Kyrgyz (NL) | 0.95 | 0.98 | 0.98 | 0.86 | 0.92 | **0.80** | **0.80** |
| Telugu (NL) | 0.99 | 1.00 | **0.63** | 0.79 | 1.00 | 0.87 | 0.87 |
| Amharic (NL) | 0.99 | 0.99 | 0.99 | **0.86** | 0.99 | 0.93 | 0.93 |
| Burmese (NL) | 0.99 | 0.99 | 0.99 | 0.94 | 0.99 | **0.83** | **0.83** |
| Nepali (NL) | 0.96 | 0.98 | **0.41** | 0.66 | 0.96 | 0.84 | 0.84 |
