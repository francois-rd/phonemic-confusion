# Load hyp_full (direct, not from data loader).
# Take N as param.
# Preprend each ref by n-1 "SOS".
# for i in range(n, len(ref)):
#  counts using i and n-1 values before i
#  numerator count is a subset of the denom count
#  find a way to count efficiently
#
# Notes:
# - No smoothing for this baseline
# - Set N in {1, 2, 3, median num phonemes/word} and see which is best.
from src.align_phonemes import ALIGNMENT_OPTS, OUTPUT_OPTS, PHONEME_OUT
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from src.utils import discrete_value_error_message as error_msg
from src.evaluation import evaluate_model
from src.data_loader import DataLoader
import collections
import pickle
import time
import os

MLE_OUT_DIR = "../output/MLE/"


def mle_baseline(data_dir, out_dir, n, weight, alignment, output, dataset,
                 cached=True):
    """
    Trains and evaluate an MLE baseline for the given parameters.

    :param data_dir: the directory from which to load the files
    :param out_dir: the directory to which to save the resulting files
    :param n: the n in n-gram
    :param weight: how much to weight mismatches over matches
    :param alignment: set to 'hypothesis'
    :param output: one of ['full', 'abridged', 'binary']
    :param dataset: one of ['val', 'test']
    :param cached: whether or not to use a cached (i.e. already trained) model
    """
    # Validate parameters.
    if alignment not in ['hypothesis']:
        raise ValueError(error_msg('alignment', ['hypothesis']))
    if output not in ['full', 'abridged', 'binary']:
        raise ValueError(error_msg('output', ['full', 'abridged', 'binary']))
    if dataset not in ['val', 'test']:
        raise ValueError(error_msg('dataset', ['val', 'test']))
    # Training.
    file = out_dir + "mle_{}_{}_{}_{}".format(n, weight, alignment, output)
    try:
        if cached:  # If should used cached model and cached model exists.
            with open(file + ".json", 'rb') as f:
                mle = pickle.load(f)  # Load it.
        else:  # Otherwise, train it from scratch and cache it for later.
            raise FileNotFoundError
    except FileNotFoundError:
        mle = _train_mle_model(data_dir, n, weight, alignment, output)
        with open(file + ".json", 'wb') as f:
            pickle.dump(mle, f)
    # Use the model to make predictions, and then evaluate the model.
    print("Loading test data...")
    dl = DataLoader(data_dir, PHONEME_OUT, alignment, output, dataset, 100000)
    print("Done.")
    refs, hyps = dl.__iter__().__next__()
    pred_hyps, pred_probs = _predict_mle_model(mle, refs, n)
    for h, hp in zip(hyps, pred_hyps):
        if len(h) != len(hp):
            print(h)
            print(hp)
    if output == 'binary':
        _save_results_for_binary(hyps, pred_hyps, pred_probs, weight, file)
    print("Evaluating...")
    start = time.time()
    evaluate_model(refs, hyps, pred_hyps, out_dir, alignment, output)
    os.rename(out_dir + alignment + "_" + output + ".txt", file + ".txt")
    print("Done. Elapsed time: {:.2f} seconds".format(time.time() - start))


def _train_mle_model(data_dir, n, weight, alignment, output):
    """
    Trains an MLE model with the given parameters.

    :param data_dir: the directory from which to load the files
    :param n: the n in n-gram
    :param weight: how much to weight mismatches over matches
    :param alignment: set to 'hypothesis'
    :param output: one of ['full', 'abridged', 'binary']
    :return: dict(tuple -> dict(int -> int)) (a representation of the MLE model)
    """
    # Validate parameters.
    if alignment not in ['hypothesis']:
        raise ValueError(error_msg('alignment', ['hypothesis']))
    if output not in ['full', 'abridged', 'binary']:
        raise ValueError(error_msg('output', ['full', 'abridged', 'binary']))
    print("Loading train data...")
    # Loading data.
    dl = DataLoader(data_dir, PHONEME_OUT, alignment, output, 'train', 100000)
    print("Done.")
    mle = collections.defaultdict(lambda: collections.defaultdict(int))
    refs, hyps = dl.__iter__().__next__()
    n_minus_1 = n - 1
    # Counting.
    refs, hyps = _prepend(refs, n_minus_1), _prepend(hyps, n_minus_1)
    for ref, hyp in zip(refs, hyps):
        for i in range(n_minus_1, len(ref)):
            value = hyp[i]
            if output == 'full':
                same = ref[i] == value
            else:
                same = value == 0
            mle[tuple(ref[i - n_minus_1:i + 1])][value] += 1 if same else weight
    # Normalizing.
    for value_dict in mle.values():
        total = sum([count for count in value_dict.values()])
        for value in value_dict:
            value_dict[value] /= total
    # Converting to normal dictionary.
    mle = dict(mle)
    mle = {k: dict(v) for k, v in mle.items()}
    return mle


def _prepend(xs, n_minus_1):
    """
    Prepends every list in the given list of lists by the given number of -1s.

    :param xs: a list of lists of integers
    :param n_minus_1: the number of -1s to prepend
    :return: the resulting list of lists on integers
    """
    prefix = [-1] * n_minus_1
    return [prefix + x for x in xs]


def _predict_mle_model(mle, refs, n):
    """
    Given an MLE model and some inputs, predict the output using MLE.

    :param mle: a trained MLE model of the form dict(tuple -> dict(int -> int))
    :param refs: inputs for prediction (list of lists of integers)
    :param n: the n in n-gram (must match the MLE model)
    :return: a list of lists of integers representing the predictions, and a
        list of prediction probabilities (only used for output == 'binary')
    """
    pred, n_minus_1, count, total = [], n - 1, 0, 0
    pred_probs = []
    print("Predicting...")
    for ref in _prepend(refs, n_minus_1):
        hyp = []
        for i in range(n_minus_1, len(ref)):
            total += 1
            try:
                value_dict = mle[tuple(ref[i - n_minus_1:i + 1])]
                hyp.append(max(value_dict.keys(), key=lambda k: value_dict[k]))
            except KeyError:
                # TODO: histogram of which n-grams causes these key errors.
                hyp.append(-1)
                count += 1
            try:
                pred_probs.append(mle[tuple(ref[i - n_minus_1:i + 1])][1])
            except KeyError:
                pred_probs.append(0)
        pred.append(hyp)
    print("Done. {} key errors of {} total key searches.".format(count, total))
    return pred, pred_probs


def _save_results_for_binary(hyps, pred_hyps, pred_probs, weight, file):
    """
    Saves a bunch of extra results in the case where output == 'binary'.

    :param hyps: the true hypotheses (list of lists of 0/1)
    :param pred_hyps: the predicted hypotheses (list of lists of 0/1)
    :param pred_probs: the probability that a prediction is 1 (list of float)
    :param weight: the weight used to train the MLE model
    :param file: the root file path for saving the resulting files
    """
    import matplotlib.pyplot as plt

    with open(file + "_extra_output.txt", 'w') as f:
        bin_hyps = [h for hyp in hyps for h in hyp]
        bin_pred_hyps = [max(0, h) for hyp in pred_hyps for h in hyp]
        tn, fp, fn, tp = confusion_matrix(bin_hyps, bin_pred_hyps).ravel()
        print("TN,{}\nFP,{}\nFN,{}\nTP,{}".format(tn, fp, fn, tp), file=f)
        print("ACC," + str(accuracy_score(bin_hyps, bin_pred_hyps)), file=f)
        fpr, sens, thresh = roc_curve(bin_hyps, pred_probs)
        print("ROC_FPR," + ','.join([str(f) for f in fpr]), file=f)
        print("ROC_SENS," + ','.join([str(s) for s in sens]), file=f)
        print("ROC_THRESHOLD," + ','.join([str(t) for t in thresh]), file=f)
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr, sens, marker='.')
        plt.title("ROC for MLE model with weight " + str(weight))
        plt.xlabel("False Positive Rate")
        plt.ylabel("Sensitivity")
        plt.savefig(file + "_ROC.pdf")
        print("ROC_AUC," + str(roc_auc_score(bin_hyps, pred_probs)), file=f)
        prec, rec, thresh = precision_recall_curve(bin_hyps, pred_probs)
        print("PRC_PREC," + ','.join([str(p) for p in prec]), file=f)
        print("PRC_REC," + ','.join([str(r) for r in rec]), file=f)
        print("PRC_THRESHOLD," + ','.join([str(t) for t in thresh]), file=f)
        plt.figure()
        plt.plot([0, 1], [0.5, 0.5], linestyle='--')
        plt.plot(rec, prec, marker='.')
        plt.title("PRC for MLE model with weight " + str(weight))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(file + "_PRC.pdf")
        print("PRC_AUC," + str(auc(rec, prec)), file=f)
    with open(file + "_predictions.txt", 'w') as f:
        for pred in pred_hyps:
            print(' '.join([str(p) for p in pred]), file=f)


if __name__ == '__main__':
    from src.phonemes import DATA_DIR

    for align in ALIGNMENT_OPTS:
        for out in OUTPUT_OPTS:
            if align != 'none' and out == 'binary':
                for w in [1, 5, 10, 20]:
                    print(align, out, w)
                    mle_baseline(DATA_DIR, MLE_OUT_DIR, 3, w, align, out, 'val',
                                 cached=False)
