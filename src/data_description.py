from src.align_phonemes import load_data, load_phonemes, text2phonemes
from src.align_phonemes import DEL, INS_SAME, INS_DIFF, HYP_CODES
from src.data_loader import DataLoader
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter
import numpy as np

FIGURES_DIR = "../output/figures/"


def to_hist(phoneme_list):
    """
    Takes a list of all phonemes in a corpus and constructs a histogram from it.

    :param phoneme_list: a list of all phonemes
    :return: labels, hist, ticks
        where labels is an ordered sequence of the phoneme labels (x-axis)
        where hist is an ordered sequence of frequency counts (y-axis)
        where ticks is simply [0, 1, 2, ..., len(labels)]
    """
    labels, hist = list(zip(*Counter(phoneme_list).most_common()))
    return labels, hist, list(range(len(hist)))


def to_csv(labels, hist, out_dir, out_file):
    """
    Outputs a CSV file of the labels and hist.

    :param labels: a sequence of labels (x-axis)
    :param hist: a sequence of frequency counts (y-axis)
    :param out_dir: the directory to which to save the resulting files
    :param out_file: the root name of the resulting files
    """
    with open(out_dir + out_file + ".csv", 'w') as f:
        for label, count in zip(labels, hist):
            print('{},{}'.format(label, count), file=f)


def plot(labels, hist, ticks, out_dir, out_file, title, y_max=None, csv=True):
    """
    Plots a histogram with the given parameters. Also outputs a CSV file of the
    labels and hist as a raw format if csv is True.

    :param labels: a sequence of labels (x-axis)
    :param hist: a sequence of frequency counts (y-axis)
    :param ticks: positional information for the labels
    :param out_dir: the directory to which to save the resulting files
    :param out_file: the root name of the resulting files
    :param title: a title for the figure
    :param y_max: (optional) the upper boundary of the y-axis
    :param csv: (optional) whether or not to output the CSV file
    """
    width, height = rcParams['figure.figsize']
    plt.figure(figsize=(width * 2, height))
    plt.bar(ticks, hist)
    if y_max is not None:
        plt.ylim(ymax=y_max)
    plt.xticks(ticks, labels)
    plt.ylabel("Frequency count")
    plt.xlabel("Phoneme")
    plt.title(title)
    plt.savefig(out_dir + out_file + ".pdf")
    if csv:
        to_csv(labels, hist, out_dir, out_file)


def phoneme_histograms(data_dir, files, phoneme_file, out_dir):
    """
    Plots histograms of phoneme frequencies in the data and records the raw data
    in CSV format.

    :param data_dir: the root directory of the transcript files
    :param files: a list of file names in the data directory
    :param phoneme_file: the file containing the phoneme dictionary
    :param out_dir: the directory to which to save the resulting files
    """
    refs, hyps = load_data(data_dir, files)
    phonemes = load_phonemes(phoneme_file)
    ref_phonemes, hyp_phonemes = [], []
    for ref, hyp in zip(refs, hyps):
        ref, hyp = text2phonemes(ref, phonemes), text2phonemes(hyp, phonemes)
        if not ref or not hyp:
            print('Skipping due to emtpy hypothesis or reference.')
            continue
        ref_phonemes.extend(ref.split())
        hyp_phonemes.extend(hyp.split())
    ref_labels, ref_hist, ref_ticks = to_hist(ref_phonemes)
    hyp_labels, hyp_hist, hyp_ticks = to_hist(hyp_phonemes)
    all_labels, all_hist, all_ticks = to_hist(ref_phonemes + hyp_phonemes)
    ymax = max(*ref_hist, *hyp_hist)
    plot(ref_labels, ref_hist, ref_ticks, out_dir, "ref",
         "Frequency of Phonemes in Reference Sentences", y_max=ymax)
    plot(hyp_labels, hyp_hist, hyp_ticks, out_dir, "hyp",
         "Frequency of Phonemes in Hypothesis Sentences", y_max=ymax)
    plot(all_labels, all_hist, all_ticks, out_dir, "all",
         "Frequency of Phonemes in the Dataset")


def full_phoneme_histogram(data_dir, out_dir):
    """
    Plots histograms of phoneme frequencies in the hypothesis_full case, and
    records the raw data in CSV format.

    :param data_dir: the root directory of the transcript files
    :param out_dir: the directory to which to save the resulting files
    """
    def replace(l, value, new_value):
        i = l.index(value)
        l.remove(value)
        l.insert(i, new_value)
    with open(data_dir + "hypothesis_full.txt", 'r') as f:
        lines = [line.strip().split('\t') for line in f]
    hyp = [p for l in lines for p in l[1].split()]
    labels, hist, ticks = to_hist(hyp)
    labels_star = list(labels)
    replace(labels_star, DEL, "*")
    replace(labels_star, INS_SAME, "**")
    replace(labels_star, INS_DIFF, "***")
    plot(labels_star, hist, ticks, out_dir, "hyp_full",
         "Frequency of Phonemes in Aligned Hypothesis Sentences", csv=False)
    to_csv(labels, hist, out_dir, "hyp_full")


def abridged_and_binary_phoneme_histograms(data_dir, out_dir):
    """
    Plots histograms of phoneme frequencies in the hypothesis_abridged and
    hypothesis_binary cases, and records the raw data in CSV format.

    :param data_dir: the root directory of the transcript files
    :param out_dir: the directory to which to save the resulting files
    """
    with open(data_dir + "hypothesis_abridged.txt", 'r') as f:
        lines = [line.strip().split('\t') for line in f]
    int_to_phoneme = {v: k for k, v in HYP_CODES.items()}
    abridged = [int_to_phoneme[int(p)] for l in lines for p in l[1].split()]
    plot(*to_hist(abridged), out_dir, "hyp_abridged",
         "Frequency of Categories in Aligned Hypothesis Sentences")
    with open(data_dir + "hypothesis_binary.txt", 'r') as f:
        lines = [line.strip().split('\t') for line in f]
    int_to_phoneme = {0: 'match', 1: 'not_match'}
    binary = [int_to_phoneme[int(p)] for l in lines for p in l[1].split()]
    plot(*to_hist(binary), out_dir, "hyp_binary",
         "Frequency of Matches in Aligned Hypothesis Sentences")


def data_summary(data_dir, files, phoneme_file, out_dir):
    """
    Records a summary of the dataset.

    :param data_dir: the root directory of the transcript files
    :param files: a list of file names in the data directory
    :param phoneme_file: the file containing the phoneme dictionary
    :param out_dir: the directory to which to save the resulting file
    """
    refs, hyps = load_data(data_dir, files)
    phonemes = load_phonemes(phoneme_file)
    ref_phonemes, hyp_phonemes = [], []
    for ref, hyp in zip(refs, hyps):
        ref, hyp = text2phonemes(ref, phonemes), text2phonemes(hyp, phonemes)
        if not ref or not hyp:
            print('Skipping due to emtpy hypothesis or reference.')
            continue
        ref_phonemes.append(len(ref.split()))
        hyp_phonemes.append(len(hyp.split()))
    with open(out_dir + "data_summary.csv", 'w') as f:
        print("num_sentences,{}".format(len(ref_phonemes)), file=f)
        x, _ = DataLoader(data_dir, phoneme_file, 'hypothesis', 'full', 'train',
                          100000).__iter__().__next__()
        print("num_train_sentences,{}".format(len(x)), file=f)
        x, _ = DataLoader(data_dir, phoneme_file, 'hypothesis', 'full', 'val',
                          100000).__iter__().__next__()
        print("num_val_sentences,{}".format(len(x)), file=f)
        x, _ = DataLoader(data_dir, phoneme_file, 'hypothesis', 'full', 'test',
                          100000).__iter__().__next__()
        print("num_test_sentences,{}".format(len(x)), file=f)
        print("avg_ref_length,{}".format(np.mean(ref_phonemes)), file=f)
        print("avg_hyp_length,{}".format(np.mean(hyp_phonemes)), file=f)


if __name__ == '__main__':
    from src.phonemes import DATA_DIR, FILES, PHONEME_OUT

    phoneme_histograms(DATA_DIR, FILES, PHONEME_OUT, FIGURES_DIR)
    full_phoneme_histogram(DATA_DIR, FIGURES_DIR)
    abridged_and_binary_phoneme_histograms(DATA_DIR, FIGURES_DIR)
    data_summary(DATA_DIR, FILES, PHONEME_OUT, FIGURES_DIR)
