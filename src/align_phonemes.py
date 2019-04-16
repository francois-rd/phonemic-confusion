from src.phonemes import DATA_DIR, FILES, PHONEME_OUT, preprocess
import numpy as np
import json

NULL, DEL, INS_SAME, INS_DIFF = 'NULL', 'DEL', 'INS_SAME', 'INS_DIFF'
ALIGNMENT_OPTS = ['none', 'hypothesis']
OUTPUT_OPTS = ['full', 'abridged', 'binary', 'scalar']
NONE_CODES = {'match': 0, 'sub': 1, 'del': 2, 'ins': 3}
HYP_CODES = {'match': 0, 'sub': 1, 'del': 2, 'ins_same': 3, 'ins_diff': 4}


def load_data(data_dir, files):
    """
    Loads all references sentences and all hypothesis sentences for all the
    given data files in the given directory. Returns two aligned lists.

    :param data_dir: the root directory of the transcript files
    :param files: a list of file names
    :return: references_list, hypothesis_list
    """
    refs, hyps = [], []
    for file in files:
        with open(data_dir + file + '.txt', 'r') as f:
            lines = [line.strip().split('\t') for line in f]
            hyps.extend([line[4].split('|')[0] for line in lines])
            refs.extend([line[-1] for line in lines])
    # Remove all empty hypotheses.
    refs, hyps = [r for r, h in zip(refs, hyps) if h], [h for h in hyps if h]
    return refs, hyps


def load_phonemes(file):
    """
    Loads and returns the phoneme dictionary from the given file. If there are
    multiple alternative phonemic transcriptions for a given word, only return
    the first one.

    :param file: the file containing the phoneme dictionary
    :return: the phoneme dictionary
    """
    with open(file, 'r') as f:
        lines = [line.strip().split('\t')[:2] for line in f]
    return {line[0].lower(): line[1] for line in lines}


def text2phonemes(text, phonemes):
    """
    Converts a string of text (whitespace-separated words) into a string of
    phonemes, based on the given phoneme dictionary.

    :param text: the text to convert
    :param phonemes: the phoneme dictionary to use
    :return: a string of phonemes
    """
    words = preprocess(text.split(), return_raw_list=True)
    return ' '.join([phonemes[word] for word in words])


def align_one(ref, hyp):
    """
    Takes the reference and hypothesis phoneme strings and minimizes the number
    of substitution, insertion, and deletion errors needed to align them (using
    a dynamic programming algorithm akin to the Levenshtein distance). Returns
    a list of pairs of phonemes representing the alignment, possibly with the
    insertion of the meta-phonemes 'DEL' and 'NULL' to ensure proper alignment.

    :param ref: a string of phonemes representing the reference sentence
    :param hyp: a string of phonemes representing the hypothesis sentence
    :return: a list of pairs of aligned phonemes
    """
    # Initialize.
    up, left, up_left, up_left_star = 0, 1, 2, 3
    n, m = len(ref), len(hyp)
    r, b = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1))
    r[0, :], b[0, :] = np.arange(m + 1), left
    r[:, 0], b[:, 0] = np.arange(n + 1), up
    b[0, 0] = -1

    # Compute forward pass.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            del_, ins = r[i - 1, j] + 1, r[i, j - 1] + 1
            sub = r[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
            r[i, j] = min(del_, sub, ins)
            if r[i, j] == del_:
                b[i, j] = up
            elif r[i, j] == sub:
                if ref[i - 1] == hyp[j - 1]:
                    b[i, j] = up_left_star
                else:
                    b[i, j] = up_left
            else:
                b[i, j] = left

    # Compute backward pass.
    i, j, alignment = n, m, []
    while i > 0 or j > 0:
        if b[i, j] == up_left or b[i, j] == up_left_star:
            i -= 1
            j -= 1
            alignment.append((ref[i], hyp[j]))
        elif b[i, j] == up:
            i -= 1
            alignment.append((ref[i], DEL))
        else:
            j -= 1
            alignment.append((NULL, hyp[j]))
    alignment.reverse()
    return alignment


class DataRecorder:
    def __init__(self, data_dir):
        """
        Modifies the alignment of given ref/hyp pairs according to each possible
        combination of alignment and output, and records these to files.

        :param data_dir: the directory in which to save the files
        """
        self.files = {}
        for alignment in ALIGNMENT_OPTS:
            for output in OUTPUT_OPTS:
                file = data_dir + alignment + "_" + output + ".txt"
                self.files[(alignment, output)] = open(file, 'w')
        self.files[('raw', 'result')] = open(data_dir + "raw_result.txt", 'w')

    def __call__(self, aligned_ref_hyp):
        """
        Modifies the alignment of given ref/hyp pairs according to each possible
        combination of alignment and output, and records these to files.

        :param aligned_ref_hyp: a list of pairs of aligned phonemes
        """
        print(json.dumps(aligned_ref_hyp), file=self.files[('raw', 'result')])
        for alignment in ALIGNMENT_OPTS:
            for output in OUTPUT_OPTS:
                self._modify_data_and_record(aligned_ref_hyp, alignment, output)

    def _modify_data_and_record(self, ref_hyp, alignment, output):
        ref = [r for r, _ in ref_hyp if r != NULL]
        if alignment == 'none':
            if output == 'full':
                hyp = ' '.join([h for _, h in ref_hyp if h != DEL])
            elif output == 'abridged':
                hyp = []
                for r, h in ref_hyp:
                    if r == h:
                        hyp.append(NONE_CODES['match'])
                    elif r == NULL:
                        hyp.append(NONE_CODES['ins'])
                    elif h == DEL:
                        hyp.append(NONE_CODES['del'])
                    else:
                        hyp.append(NONE_CODES['sub'])
                hyp = ' '.join([str(h) for h in hyp])
            elif output == 'binary':
                hyp = ' '.join([str(int(r != h)) for r, h in ref_hyp])
            else:  # output == 'scalar'
                hyp = str(np.count_nonzero([r != h for r, h in ref_hyp]))
        else:  # alignment == 'hypothesis'
            hyp = []
            start = 0
            while ref_hyp[start][0] == NULL:
                start += 1
            for i in range(len(ref_hyp)):
                (r, h), not_last = ref_hyp[i], i < len(ref_hyp) - 1
                if r == NULL:
                    continue
                elif i == start > 0:
                    hyp.append(INS_SAME if r == h else INS_DIFF)
                elif r != 'NULL' and not_last and ref_hyp[i + 1][0] == NULL:
                    hyp.append(INS_SAME if r == h else INS_DIFF)
                else:
                    hyp.append(h)
            if output == 'full':
                hyp = ' '.join([str(h) for h in hyp])
            elif output == 'abridged':
                hyp_tmp = []
                for r, h in zip(ref, hyp):
                    if r == h:
                        hyp_tmp.append(HYP_CODES['match'])
                    elif h == INS_SAME:
                        hyp_tmp.append(HYP_CODES['ins_same'])
                    elif h == INS_DIFF:
                        hyp_tmp.append(HYP_CODES['ins_diff'])
                    elif h == DEL:
                        hyp_tmp.append(HYP_CODES['del'])
                    else:
                        hyp_tmp.append(HYP_CODES['sub'])
                hyp = ' '.join([str(h) for h in hyp_tmp])
            elif output == 'binary':
                hyp = ' '.join([str(int(r != h)) for r, h in zip(ref, hyp)])
            else:  # output == 'scalar'
                hyp_tmp = []
                for r, h in zip(ref, hyp):
                    if r == h:
                        hyp_tmp.append(0)
                    elif h == INS_DIFF:
                        hyp_tmp.append(2)
                    else:
                        hyp_tmp.append(1)
                hyp = str(np.sum(hyp_tmp))
        print(' '.join(ref) + "\t" + hyp, file=self.files[(alignment, output)])

    def close(self):
        """
        Closes all open files.
        """
        for file in self.files.values():
            file.close()


def align_all(refs, hyps, phonemes, data_dir, verbose=True):
    """
    For each pair of lines in refs and hyps, convert all words to phonemes,
    align their individual phonemes using all variations of alignment/output
    combinations, and then record the results in new files.

    :param refs: a list of reference sentences
    :param hyps: a list of hypothesis sentences
    :param phonemes: the phoneme dictionary to use
    :param data_dir: the directory in which to save the files
    :param verbose: whether to print progress information
    """
    data_recorder = DataRecorder(data_dir)
    i, n = 0, len(refs)
    for ref, hyp in zip(refs, hyps):
        if i % (n // min(10, n)) == 0 and verbose:
            print("Aligned {} of {}".format(i, n))
        i += 1
        ref, hyp = text2phonemes(ref, phonemes), text2phonemes(hyp, phonemes)
        if not ref or not hyp:
            print('Skipping due to emtpy hypothesis or reference.')
            continue
        data_recorder(align_one(ref.split(), hyp.split()))
    data_recorder.close()


if __name__ == '__main__':
    align_all(*load_data(DATA_DIR, FILES), load_phonemes(PHONEME_OUT), DATA_DIR)
