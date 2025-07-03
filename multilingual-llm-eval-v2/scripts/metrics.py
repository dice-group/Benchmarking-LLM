from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics import CHRF
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer

def compute_em(predictions, references):
    exact_matches = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references))
    return 100.0 * exact_matches / len(predictions)

def compute_bleu(predictions, references):
    smoothie = SmoothingFunction().method4
    scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie) for pred, ref in zip(predictions, references)]
    return 100.0 * sum(scores) / len(scores)

def compute_chrf(predictions, references):
    chrf = CHRF()
    return chrf.corpus_score(predictions, [references]).score

def compute_f1(predictions, references):
    def tokenize(text):
        return text.lower().split()

    all_preds = [tokenize(p) for p in predictions]
    all_refs = [tokenize(r) for r in references]

    label_set = list(set(token for tokens in all_preds + all_refs for token in tokens))
    label_index = {label: i for i, label in enumerate(label_set)}

    def vectorize(tokens):
        vec = [0] * len(label_set)
        for token in tokens:
            if token in label_index:
                vec[label_index[token]] = 1
        return vec

    y_pred = [vectorize(p) for p in all_preds]
    y_true = [vectorize(r) for r in all_refs]

    return 100.0 * f1_score(y_true, y_pred, average="micro")

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]

    rouge1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
    rougeL = sum(s['rougeL'].fmeasure for s in scores) / len(scores)

    return {"ROUGE-1": 100.0 * rouge1, "ROUGE-L": 100.0 * rougeL}

def compute_accuracy(predictions, references):
    correct = sum(int(p == r) for p, r in zip(predictions, references))
    return 100.0 * correct / len(references)

def compute_all_metrics(predictions, references):
    rouge = compute_rouge(predictions, references)
    return {
        "EM": compute_em(predictions, references),
        "F1": compute_f1(predictions, references),
        "BLEU": compute_bleu(predictions, references),
        "chrF": compute_chrf(predictions, references),
        "acc": compute_accuracy(predictions, references),
        **rouge
    }
