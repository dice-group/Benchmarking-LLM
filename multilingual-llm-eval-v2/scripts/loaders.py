import json
import os
from datasets import load_dataset

def load_dataset_local(data_dir, task, lang):
    file_path = os.path.join(data_dir, f"{lang}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if task == "qa":
        return [(d["context"], d["question"], d["answer"]) for d in data]
    elif task == "generation":
        return [(d["context"], None, d["answer"]) for d in data]
    elif task == "translation":
        return [(d["source"], None, d["target"]) for d in data]
    elif task == "summarization":
        return [(d["document"], None, d["summary"]) for d in data]
    else:
        raise ValueError(f"Unsupported task: {task}")

def load_culturax(lang, task, max_samples=10):
    hf_ds = load_dataset("uonlp/CulturaX", lang, split="train")
    samples = hf_ds.shuffle(seed=42).select(range(max_samples))["text"]
    return [(text, None, text) for text in samples]


def load_belebele(language_code="eng_Latn", split="validation", sample_size=1000):
    dataset = load_dataset("facebook/belebele", name=language_code, split=split)
    dataset = dataset.select(range(min(sample_size, len(dataset))))
    
    inputs = []
    references = []
    
    for example in dataset:
        prompt = f"{example['context']}\n\nQuestion: {example['question']}\n"
        for i, option in enumerate(example['mc_answers']['choices']):
            prompt += f"{chr(65 + i)}. {option}\n"
        prompt += "\nAnswer:"
        
        inputs.append(prompt)
        references.append(example['mc_answers']['answer'])

    return inputs, references