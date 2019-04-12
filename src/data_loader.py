from src.utils import discrete_value_error_message as error_msg
from src.align_phonemes import ALIGNMENT_OPTS, OUTPUT_OPTS
from src.align_phonemes import DEL, INS_SAME, INS_DIFF
from src.align_phonemes import NONE_CODES, HYP_CODES
from sklearn.model_selection import train_test_split
from src.phonemes import PHONEME_OUT, DATA_DIR


class DataLoader:
    def __init__(self, data_dir, phoneme_file, alignment, output, dataset,
                 batch_size):
        """
        Loads an aligned dataset from the given directory, converting phonemes
        to integers where necessary. This class is an iterator. Each __next__()
        returns (X, y), where X and y are both lists.
            X[i] is a list of integers representing the phoneme sequence in
                reference sentence i
            y[i] contains the output associated with reference sentence i
                if output == 'full' this is a list of integers representing the
                    phoneme sequence in hypothesis i
                if output == 'abridged' this is a list of integers representing
                    an abridged version of the phoneme sequence in hypothesis i
                if output == 'binary' this is a list of 1s or 2s representing
                    match/mismatch in the alignment of ref i and hyp i
                if output == 'scalar' this is a single integer representing the
                    number of 2s that would be in the output == 'binary' list
        The returned values are mini-batches, rather than the full dataset.
        These can be either from the training, validation, or testing
        distributions.

        :param data_dir: the directory from which to load the files
        :param phoneme_file: the file containing the phoneme dictionary
        :param alignment: one of ['none', 'hypothesis']
        :param output: one of ['full', 'abridged', 'binary', 'scalar']
        :param dataset: one of ['train', 'val', 'test']
        :param batch_size: the mini-batch size
        """
        # Valdiate parameters.
        if alignment not in ALIGNMENT_OPTS:
            raise ValueError(error_msg('alignment', ALIGNMENT_OPTS))
        if output not in OUTPUT_OPTS:
            raise ValueError(error_msg('output', OUTPUT_OPTS))
        if dataset not in ['train', 'val', 'test']:
            raise ValueError(error_msg('dataset', ['train', 'val', 'test']))
        # Load phoneme2int dictionary.
        with open(phoneme_file, 'r') as f:
            lines = [line.strip().split('\t')[1] for line in f]
            if alignment == 'hypothesis':
                lines.append('{} {} {}'.format(DEL, INS_SAME, INS_DIFF))
            phonemes = sorted(set([p for line in lines for p in line.split()]))
        phoneme_to_int = {p: i + 1 for i, p in enumerate(phonemes)}
        # Create the int2phoneme dictionary.
        if output == 'binary':
            self.int_to_phoneme = {1: 'match', 2: 'not_match'}
        elif output == 'abridged':
            if alignment == 'none':
                self.int_to_phoneme = {v: k for k, v in NONE_CODES}
            else:
                self.int_to_phoneme = {v: k for k, v in HYP_CODES}
        elif output == 'scalar':
            self.int_to_phoneme = None
        else:
            self.int_to_phoneme = {v: k for k, v in phoneme_to_int}
        # Load dataset.
        with open(data_dir + alignment + "_" + output + ".txt", 'r') as f:
            lines = [line.strip().split('\t') for line in f]
        ref = [[phoneme_to_int[p] for p in l[0].split()] for l in lines]
        if output == 'full':
            hyp = [[phoneme_to_int[p] for p in l[1].split()] for l in lines]
            self.vocab_size = len(phoneme_to_int) + 1
        elif output == 'binary' or output == 'abridged':
            hyp = [[int(b) for b in l[1].split()] for l in lines]
            if output == 'binary':
                self.vocab_size = 3
            elif alignment == 'none':
                self.vocab_size = len(NONE_CODES) + 1
            else:
                self.vocab_size = len(HYP_CODES) + 1
        else:  # output == 'scalar'
            hyp = [int(s[1]) for s in lines]
            self.vocab_size = 1
        # Prune dataset.
        ref_train, ref_test, hyp_train, hyp_test = train_test_split(
            ref, hyp, test_size=0.25, random_state=0)
        ref_val, ref_test, hyp_val, hyp_test = train_test_split(
            ref_test, hyp_test, test_size=0.5, random_state=0)
        if dataset == 'train':
            self.ref, self.hyp = ref_train, hyp_train
        elif dataset == 'val':
            self.ref, self.hyp = ref_val, hyp_val
        else:  # dataset == 'test'
            self.ref, self.hyp = ref_test, hyp_test
        # Prepare for iterations.
        self.i, self.random_state, self.size = 0, 0, len(self.ref)
        self.batch_size = batch_size

    def __iter__(self):
        self.iter = range(0, self.size, self.batch_size).__iter__()
        self.random_state += 1
        self.ref, _, self.hyp, _ = train_test_split(
            self.ref, self.hyp, test_size=0, random_state=self.random_state)
        return self

    def __next__(self):
        i = self.iter.__next__()
        return self.ref[i:i + self.batch_size], self.hyp[i:i + self.batch_size]


if __name__ == '__main__':
    for align in ALIGNMENT_OPTS:
        for out in OUTPUT_OPTS:
            for data in ['train', 'val', 'test']:
                dl = DataLoader(DATA_DIR, PHONEME_OUT, align, out, data, 256)
                data_check = {0: {'X': [], 'y': []}, 1: {'X': [], 'y': []}}
                for epoch in range(2):
                    for X, y in dl:
                        data_check[epoch]['X'].extend(X)
                        data_check[epoch]['y'].extend(y)
                assert len(data_check[0]['X']) == len(data_check[1]['X'])
                assert len(data_check[0]['y']) == len(data_check[1]['y'])
                assert len(data_check[0]['X']) == len(data_check[0]['y'])
                print(align, out, data, len(data_check[0]['X']))
