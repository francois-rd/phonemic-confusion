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
from src.utils import discrete_value_error_message as error_msg
from src.evaluation import evaluate_model
from src.data_loader import DataLoader
import collections
import pickle
import time

MLE_OUT_DIR = "../output/MLE/"


def mle_baseline(data_dir, out_dir, n, alignment, output, dataset, cached=True):
    """
    Trains and evaluate an MLE baseline for the given parameters.

    :param data_dir: the directory from which to load the files
    :param out_dir: the directory to which to save the resulting files
    :param n: the n in n-gram
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
    file = out_dir + "mle_" + alignment + "_" + output + ".json"
    try:
        if cached:  # If should used cached model and cached model exists.
            with open(file, 'rb') as f:
                mle = pickle.load(f)  # Load it.
        else:  # Otherwise, train it from scratch and cache it for later.
            raise FileNotFoundError
    except FileNotFoundError:
        mle = _train_mle_model(data_dir, n, alignment, output)
        with open(file, 'wb') as f:
            pickle.dump(mle, f)
    # Use the model to make predictions, and then evaluate the model.
    print("Loading test data...")
    dl = DataLoader(data_dir, PHONEME_OUT, alignment, output, dataset, 100000)
    print("Done.")
    refs, hyps = dl.__iter__().__next__()
    pred_hyps = _predict_mle_model(mle, refs, n)
    print("Evaluating...")
    start = time.time()
    evaluate_model(refs, hyps, pred_hyps, out_dir, alignment, output)
    print("Done. Elapsed time: {:.2f} seconds".format(time.time() - start))


def _train_mle_model(data_dir, n, alignment, output):
    """
    Trains an MLE model with the given parameters.

    :param data_dir: the directory from which to load the files
    :param n: the n in n-gram
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
            mle[tuple(ref[i - n_minus_1:i + 1])][hyp[i]] += 1
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
    :return: a list of lists of integers representing the predictions
    """
    pred, n_minus_1, count, total = [], n - 1, 0, 0
    print("Predicting...")
    for ref in _prepend(refs, n_minus_1):
        hyp = []
        for i in range(n_minus_1, len(ref)):
            total += 1
            try:
                value_dict = mle[tuple(ref[i - n_minus_1:i + 1])]
                hyp.append(max(value_dict.keys(), key=lambda k: value_dict[k]))
            except KeyError:
                # Naively skip the output if the context was never seen.
                # TODO: histogram of which n-grams causes these key errors.
                hyp.append(-1)
                count += 1
        pred.append(hyp)
    print("Done. {} key errors of {} total key searches".format(count, total))
    return pred


if __name__ == '__main__':
    from src.phonemes import DATA_DIR

    for align in ALIGNMENT_OPTS:
        for out in OUTPUT_OPTS:
            if align != 'none' and out != 'scalar':
                print(align, out)
                mle_baseline(DATA_DIR, MLE_OUT_DIR, 3, align, out, 'val',
                             cached=False)
