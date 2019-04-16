from src.align_phonemes import align_one, HYP_CODES, NONE_CODES
import numpy as np


def lev_from_alignment(x_y_alignment_list, lengths):
    """
    Given a list of aligned pairs and a list of lengths, computes the
    Levenshtein for each pair in the list. Returns a list of distances.

    :param x_y_alignment_list: a list where each element is itself a list
        containing tuples representing the alignment of some x with some y
    :param lengths: a numpy array of lengths for normalization
    :return: a list of Levenshtein distances
    """
    dists = []
    for x_y_alignment in x_y_alignment_list:
        dists.append(np.count_nonzero([x != y for x, y in x_y_alignment]))
    return np.array(dists) / lengths


def compute_confusion(xs, ys, lengths, alignment, output):
    """
    Given a list of inputs xs, a list of outputs xs, and a list of lengths,
    computes the confusion between each x and each y according to the given
    alignment and output scheme. Returns a list of confusion scores.

    :param xs: a list of lists of ints representing the inputs
    :param ys: a list of lists of ints representing the outputs
    :param lengths: a numpy array of lengths for normalization
    :param alignment: one of ['none', 'hypothesis']
    :param output: one of ['full', 'abridged', 'binary', 'scalar']
    :return: a list of confusion scores
    """
    if output == 'full':
        if alignment == 'none':
            alignments = [align_one(*x_y) for x_y in zip(xs, ys)]
        else:
            alignments = [list(zip(*x_y)) for x_y in zip(xs, ys)]
        return lev_from_alignment(alignments, lengths)
    elif output == 'abridged':
        if alignment == 'none':
            dists, match = [], NONE_CODES['match']
            for y in ys:
                dists.append(np.count_nonzero([match != y_ for y_ in y]))
            return np.array(dists) / lengths
        else:
            dists = []
            for y in ys:
                count = 0
                for y_ in y:
                    if y_ == HYP_CODES['match']:
                        continue
                    elif y_ == HYP_CODES['ins_diff']:
                        count += 2
                    else:
                        count += 1
                dists.append(count)
            return np.array(dists) / lengths
    elif output == 'binary':
        dists, match = [], 0
        for y in ys:
            dists.append(np.count_nonzero([match != y_ for y_ in y]))
        return np.array(dists) / lengths
    else:  # output == 'scalar'
        return np.array(ys) / lengths


def evaluate_model(x, y_true, y_pred, out_dir, alignment, output):
    """
    Given the groud truth input (x), the ground truth output (y_true), and the
    output predicted by some model (y_pred), records a file containing a matrix
    with len(x) number of rows and 3 columns. Each row represents one entry from
    x, y_true, and y_pred. The columns are as follows:
        Column 1: the ground truth confusion between x and y_true
        Column 2: the predicted confusion between x and y_pred
        Column 3:
            if output == 'scalar', this is the absolute error between each entry
                of y_true and y_pred
            else, this is the Levenshtein distance between y_true and y_pred
    :param x: a list of lists of ints representing the inputs
    :param y_true: a list of lists of ints representing the true outputs
    :param y_pred: a list of lists of ints representing the predicted outputs
    :param out_dir: the directory to which to save the resulting file
    :param alignment: one of ['none', 'hypothesis']
    :param output: one of ['full', 'abridged', 'binary', 'scalar']
    """
    # Compute the confusion between (x and y_true) and (x and y_pred).
    lengths = np.array([len(x_) for x_ in x])
    true_confusion = compute_confusion(x, y_true, lengths, alignment, output)
    pred_confusion = compute_confusion(x, y_pred, lengths, alignment, output)

    # Compute the distance between y_true and y_pred as another way to evaluate
    # how far off the prediction was from the ground truth. This is called
    # output deviation. For scalar output, it is just the error. For all others,
    # it is the Lev distance between y_true and y_pred, normalized by y_true.
    if output == 'scalar':
        output_deviation = np.abs(np.array(y_true) - np.array(y_pred))
    else:
        lengths = np.array([len(y) for y in y_true])
        output_alignment = [align_one(*y_y) for y_y in zip(y_true, y_pred)]
        output_deviation = lev_from_alignment(output_alignment, lengths)

    # Record the evaluation results to file.
    result = np.column_stack((true_confusion, pred_confusion, output_deviation))
    file = out_dir + alignment + "_" + output + ".txt"
    np.savetxt(file, result, delimiter=',', header='gt_conf,pred_conf,out_dev')


if __name__ == '__main__':
    from src.align_phonemes import ALIGNMENT_OPTS, OUTPUT_OPTS

    ref = [[1, 2, 3, 4], [3, 2, 2]]
    for align in ALIGNMENT_OPTS:
        for out in OUTPUT_OPTS:
            if align == 'none':
                if out == 'full':
                    hyp_true = [[1, 3, 2, 4, 5, 6], [3]]
                    hyp_pred = [[1, 2, 3, 4, 5, 6], [3, 2]]
                elif out == 'abridged':
                    hyp_true = [[0, 1, 1, 0, 3, 3], [0]]
                    hyp_pred = [[0, 0, 0, 0, 3, 3], [0, 0]]
                elif out == 'binary':
                    hyp_true = [[0, 1, 1, 0, 1, 1], [0]]
                    hyp_pred = [[0, 0, 0, 0, 1, 1], [0, 0]]
                else:  # out == 'scalar'
                    hyp_true = [4, 1]
                    hyp_pred = [2, 0]
            else:
                if out == 'full':
                    hyp_true = [[1, 3, 2, 10], [3, 11, 11]]
                    hyp_pred = [[1, 2, 3, 4], [3, 2, 11]]
                elif out == 'abridged':
                    hyp_true = [[0, 1, 1, 3], [0, 2, 2]]
                    hyp_pred = [[0, 0, 0, 3], [0, 0, 2]]
                elif out == 'binary':
                    hyp_true = [[0, 1, 1, 1], [0, 1, 1]]
                    hyp_pred = [[0, 0, 0, 1], [0, 0, 1]]
                else:  # out == 'scalar'
                    hyp_true = [3, 2]
                    hyp_pred = [1, 1]
            evaluate_model(ref, hyp_true, hyp_pred, '../output/', align, out)
