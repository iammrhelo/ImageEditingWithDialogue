import math
import pickle
import sys
from collections import Counter
from itertools import chain

import numpy as np


class VocabEntry(object):
    def __init__(self):
        self.word2id = {}
        self.word2id['<pad>'] = self.pad_id = 0
        self.word2id['<unk>'] = self.unk_id = 1

        self._id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self._id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self._id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, indices):
        if len(indices) == 0:
            return []
        elif type(indices[0]) == list:
            return [self.indices2words(i) for i in indices]
        else:
            return [self.id2word(i) for i in indices]

    @classmethod
    def from_corpus(self, corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types:', len(word_freq),
              ', >= freq:', len(valid_words))

        top_k_words = sorted(
            valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry


class Vocab(object):
    def __init__(self, sentences, tags, intents, size, cutoff):

        self.word = VocabEntry.from_corpus(sentences, size, cutoff)
        self.tag = VocabEntry.from_corpus(tags, size, 0)
        self.intent = VocabEntry.from_corpus([intents], size, 0)

    def __repr__(self):
        return 'Vocab(word: {}, tag: {}, intent: {})'\
            .format(len(self.word), len(self.tag), len(self.intent))


def parse_row(line):
    """ Parse row into x, y information
    """
    line = line.strip()

    word_tags, intent = line.split('|')

    words = []
    tag = []
    for word_tag in word_tags.split():
        word, word_tag = word_tag.split("###")
        words.append(word)
        tag.append(word_tag)

    return words, tag, intent


def read_tag_file(file_path):
    """  Reads tag file
    Args:
        file_path (str)
    Returns:
        sentences (list)
        tags (list)
        intent (list)
    """
    sentences = []
    tags = []
    intents = []

    with open(file_path, 'r') as fin:
        next(fin)  # Skip header
        for line in fin.readlines():
            words, tag, intent = parse_row(line)

            sentences.append(words)
            tags.append(tag)
            intents.append(intent)

    return sentences, tags, intents


def load_from_pickle(file_path):
    with open(file_path, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        sents, tags, intents = list(zip(*examples))

        yield sents, tags, intents


def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) >
                        i else pad_token for k in range(batch_size)])

    return sents_t


"""
Code taken from https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py
Include original comments
"""

# compute f1 score is modified from conlleval.pl


def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart


def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd


def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType


def computeF1Score(correct_slots, pred_slots):
    correctChunk = {}
    correctChunkCnt = 0
    foundCorrect = {}
    foundCorrectCnt = 0
    foundPred = {}
    foundPredCnt = 0
    correctTags = 0
    tokenCount = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                   __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                   (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1
                    else:
                        correctChunk[lastCorrectType] = 1
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                        (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
               __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
               (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1
                else:
                    foundCorrect[correctType] = 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1
                if predType in foundPred:
                    foundPred[predType] += 1
                else:
                    foundPred[predType] = 1

            if correctTag == predTag and correctType == predType:
                correctTags += 1

            tokenCount += 1

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1
            else:
                correctChunk[lastCorrectType] = 1

    if foundPredCnt > 0:
        precision = 100*correctChunkCnt/foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 100*correctChunkCnt/foundCorrectCnt
    else:
        recall = 0

    if (precision+recall) > 0:
        f1 = (2*precision*recall)/(precision+recall)
    else:
        f1 = 0

    return f1, precision, recall
