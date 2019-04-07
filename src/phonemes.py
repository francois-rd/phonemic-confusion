from bs4 import BeautifulSoup, Comment
from num2words import num2words
from string import punctuation
import collections
import requests
import inflect
import re
import os

DATA_DIR = "../data/transcripts/"
PHONEME_OUT = '../data/phonemes.txt'
FILES = ['huck', 'treasure', 'emma', 'sherlock']
MAX_SIZE = 10000  # Size of word files to send to CMU website.


def unique(data_dir, files):
    """
    Generate set of unique words used in both transcriptions and ground truth.

    :param data_dir: the root directory of the transcript files
    :param files: a list of file names
    :return: a set of unique words in all files.
    """
    words = []
    for file in files:
        with open(data_dir + file + '.txt', 'r') as f:
            # Get all lines except the first (empty line).
            lines = [line.strip().split('\t') for line in f][1:]
            # Add transcriptions words.
            words.extend([word for line in lines for word in
                          line[4].replace(' ', '|').split('|')])
            # Add ground truth words.
            words.extend([word for line in lines for word in line[-1].split()])
    return set(words)


def number2words(inflect_engine, candidate):
    """
    Spells out a string representation of a number with digits, or leaves the
    word unchanged if it does not contain a digit.
        e.g. '$6,000' => ['six', 'thousand', 'dollars']
        e.g. 'apple' => ['apple']

    :param inflect_engine: inflect.engine()
    :param candidate: a candidate word
    :return: the written form of a number, as a list of words
        e.g. '$6,000' => ['six', 'thousand', 'dollars']
        e.g. 'apple' => ['apple']
    """
    result = candidate
    if re.search(r'\d', candidate):  # Candidate contains digit.
        if re.search(r'\$.+\..+', candidate):  # Form: $xxx.xxx
            result = num2words(candidate[1:], to='currency')
            if re.search(r'\$1\..+', candidate):  # Form: $1.xxx
                result = result.replace('euro,', 'dollar and')
            else:
                result = result.replace('euro,', 'dollars and')
        else:
            result = inflect_engine.number_to_words(candidate)
            if result.startswith('$'):  # From: $xxxx
                if "$1" == candidate:
                    result += 'dollar'
                else:
                    result += "dollars"
    return result.split()


def preprocess(words):
    """
    Converts all numeric strings into words (e.g. '1' => 'one), removes all
    punctuation, removes all whitespace-only words, and returns a sorted set
    of the resulting words.

    :param words: the set of words to preprocess
    :return: a sorted set of preprocessed words
    """
    p = inflect.engine()
    words = set([x for w in words for x in number2words(p, w)])
    # Remove all punctuation.
    words = [w.translate(str.maketrans('', '', punctuation)) for w in words]
    # Remove all entries with whitespace only and create a sorted set.
    return sorted(set([w for w in words if len(w.strip()) > 0]))


def write_phonemes(words, size, file):
    """
    For each word in the given set of words, use the CMU phoneme dictionary to
    find all phonemic transcriptions of the word, and write them to the given
    file.

    :param words: the set of words to process
    :param size: max number of words to send to the CMU website in one chunk
    :param file: the file to which to write the results
    """
    result = collections.defaultdict(list)
    num = len(list(range(0, len(words), size)))
    for i in range(0, len(words), size):
        # Save a chunk to file.
        with open('to_send.txt', 'w') as f:
            print('\n'.join(words[i:i + size]), file=f)
        # Upload file.
        r = requests.post(
            'http://www.speech.cs.cmu.edu//cgi-bin/tools/logios/lextool.pl',
            files={'wordfile': ('to_send.txt', open('to_send.txt', 'rb'))}
        )
        # Find URL of output and capture results.
        soup = BeautifulSoup(r.text, 'lxml')
        url = soup.findAll(text=lambda text: isinstance(text, Comment))
        output = requests.get(url[0].split(' ')[2]).text.split('\n')
        # Create/update dictionary with alternative phoneme combinations.
        for line in output[:-1]:
            line = line.split('\t')
            result[line[0].split('(')[0]].append(line[1])
        print('Chunk {} of {} completed.'.format(i // size + 1, num))
    os.remove('to_send.txt')
    # Add dictionary to file.
    with open(file, 'w') as f:
        for word, phonemes in result.items():
            print('\t'.join([word] + phonemes), file=f)


if __name__ == '__main__':
    write_phonemes(preprocess(unique(DATA_DIR, FILES)), MAX_SIZE, PHONEME_OUT)
