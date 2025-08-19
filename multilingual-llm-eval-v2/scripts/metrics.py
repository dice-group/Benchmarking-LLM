from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics import CHRF
from sklearn.metrics import f1_score
#from rouge_score import rouge_scorer

def compute_em(predictions, references):
    """Exact Match accuracy in percentage."""
    assert len(predictions) == len(references), "Predictions and references must have same length"
    exact_matches = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references))
    return 100.0 * exact_matches / len(predictions)

def compute_bleu(predictions, references):
    """Corpus-level BLEU score in percentage."""
    smoothie = SmoothingFunction().method4
    scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie) 
              for pred, ref in zip(predictions, references)]
    return 100.0 * sum(scores) / len(scores)

def compute_chrf(predictions, references):
    """chrF (character n-gram F-score) - character level only."""
    chrf = CHRF(word_order=0)
    refs_nested = [[ref] for ref in references]
    return chrf.corpus_score(predictions, refs_nested).score

def compute_chrfpp(predictions, references):
    """chrF++ (character + word n-gram F-score)."""
    chrfpp = CHRF()
    refs_nested = [[ref] for ref in references]
    return chrfpp.corpus_score(predictions, refs_nested).score

def compute_f1(predictions, references):
    """Token-level micro-averaged F1 score in percentage."""
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

#def compute_rouge(predictions, references):
#    """ROUGE-1 and ROUGE-L F-measure scores in percentage."""
#    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
#
#    rouge1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
#    rougeL = sum(s['rougeL'].fmeasure for s in scores) / len(scores)
#
#    return {"ROUGE-1": 100.0 * rouge1, "ROUGE-L": 100.0 * rougeL}

def compute_accuracy(predictions, references):
    """Simple accuracy in percentage."""
    assert len(predictions) == len(references), "Predictions and references must have same length"
    correct = sum(int(p == r) for p, r in zip(predictions, references))
    return 100.0 * correct / len(references)

def compute_all_metrics(predictions, references):
    rouge = compute_rouge(predictions, references)
    return {
        "EM": compute_em(predictions, references),
        "F1": compute_f1(predictions, references),
        "BLEU": compute_bleu(predictions, references),
        "chrF": compute_chrf(predictions, references),
        "chrF++": compute_chrfpp(predictions, references),
        "acc": compute_accuracy(predictions, references),
        **rouge
    }

