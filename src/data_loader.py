from src.utils import discrete_value_error_message as error_msg
from src.align_phonemes import ALIGNMENT_OPTS, OUTPUT_OPTS
from src.phonemes import PHONEME_OUT, DATA_DIR


def phoneme2int(file):
    """
    Returns a dictionary mapping each phoneme, including the meta-phonemes, to
    a unique integer, which can then be used for one-hot encoding.

    :param file: the file containing the phoneme dictionary
    :return: a dictionary mapping each phoneme to a unique integer
    """
    with open(file, 'r') as f:
        lines = [line.strip().split('\t')[1] for line in f]
        lines.append('DEL INS')
        phonemes = sorted(set([p for line in lines for p in line.split()]))
    return {p: i for i, p in enumerate(phonemes)}


class DataLoader:
    def __init__(self, data_dir, phoneme_to_int):
        """
        Loads an aligned dataset from the given directory, converting phonemes
        to integers where necessary.

        :param data_dir: the directory from which to load the files
        :param phoneme_to_int: a dictionary mapping phonemes to unique integers
        """
        self.data_dir, self.phoneme2int = data_dir, phoneme_to_int

    def __call__(self, alignment, output):
        """
        Loads the dataset with the given alignment and output combination.
        Returns (X, y), where X and y are both lists.
            X[i] is a list of integers representing the phoneme sequence in
                reference sentence i
            y[i] contains the output associated with reference sentence i
                if output == 'full' this is a list of integers representing the
                    phoneme sequence in hypothesis i
                if output == 'binary' this is a list of 0s or 1s representing
                    match/mismatch in the alignment of ref i and hyp i
                if output == 'scalar' this is a single integer representing the
                    number of 1s that would be in the output == 'binary' list
        :param alignment: one of ['none', 'hypothesis']
        :param output: one of ['full', 'binary', 'scalar']
        :return: list[list[int]], list[Union[list[int], int]]
        """
        if alignment not in ALIGNMENT_OPTS:
            raise ValueError(error_msg('alignment', ALIGNMENT_OPTS))
        if output not in OUTPUT_OPTS:
            raise ValueError(error_msg('output', OUTPUT_OPTS))
        with open(self.data_dir + alignment + "_" + output + ".txt", 'r') as f:
            lines = [line.strip().split('\t') for line in f]
        ref = [[self.phoneme2int[p] for p in l[0].split()] for l in lines]
        if output == 'full':
            for l in lines:
                if len(l) != 2:
                    print(l)
            hyp = [[self.phoneme2int[p] for p in l[1].split()] for l in lines]
        elif output == 'binary':
            hyp = [[int(b) for b in l[1].split()] for l in lines]
        else:  # output == 'scalar'
            hyp = [int(s[1]) for s in lines]
        return ref, hyp


if __name__ == '__main__':
    dl = DataLoader(DATA_DIR, phoneme2int(PHONEME_OUT))
    for align in ALIGNMENT_OPTS:
        for out in OUTPUT_OPTS:
            print(dl(align, out))
