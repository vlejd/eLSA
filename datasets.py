"""Classes for dataset manipulation.
First download the https://github.com/facebookresearch/SentEval.git repository and download all data files according to the manual.
Then set environment variable SENTEVAL_DATA_BASE to path to the data directories.
"""
import os, io
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


SENTEVAL_DATA_BASE = os.environ['SENTEVAL_DATA_BASE']


class Dataset(object):
    def __init__(self, seed=1111):
        self.seed = seed
        self.positives = self.load_positives()
        self.negatives = self.load_negatives()
        self.samples = self.positives + self.negatives
        self.labels = np.array([1] * len(self.positives) + [0] * len(self.negatives))
        self.n_samples = len(self.samples)
        self.reshufle(seed)

    def reshufle(self, random_state=None, folds=10):
        train_shuffle = StratifiedShuffleSplit(folds, random_state=random_state, test_size=0.2)
        generator1 = train_shuffle.split(self.samples, self.labels)
        for split_index, (self.train_ids, others) in enumerate(generator1):
            test_shuffle = StratifiedShuffleSplit(folds, random_state=random_state, test_size=0.5)
            generator2 = test_shuffle.split(others, self.labels[others])
            other_valid_id, other_test_id = next(iter(generator2))
            self.valid_ids, self.test_ids = others[other_valid_id], others[other_test_id]
            yield split_index

    def train_samples(self):
        return np.array(self.samples)[self.train_ids]

    def train_labels(self):
        return self.labels[self.train_ids]

    def valid_samples(self):
        return np.array(self.samples)[self.valid_ids]

    def valid_labels(self):
        return self.labels[self.valid_ids]

    def test_samples(self):
        return np.array(self.samples)[self.test_ids]

    def test_labels(self):
        return self.labels[self.test_ids]

    def loadFile(self, fpath):
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line.split() for line in f.read().splitlines()]

    def baseline(self):
        mean = np.mean(self.labels)
        return max(mean, 1-mean)

    def name(self):
        return self.__class__.__name__

    def bias(self):
        mean = np.mean(self.labels)
        return max(mean, 1-mean)

class CRDataset(Dataset):
    def load_positives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'CR/custrev.pos'))

    def load_negatives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'CR/custrev.neg'))

class MRDataset(Dataset):
    def load_positives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'MR/rt-polarity.pos'))

    def load_negatives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'MR/rt-polarity.neg'))

class SUBJDataset(Dataset):
    def load_positives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'SUBJ/subj.objective'))

    def load_negatives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'SUBJ/subj.subjective'))

class MPQADataset(Dataset):
    def load_positives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'MPQA/mpqa.pos'))

    def load_negatives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'MPQA/mpqa.neg'))

class TRECDataset(Dataset):
    SUPPORTED_LABELS = set(['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM'])

    def __init__(self, seed=1111, task_label='ABBR'):
        self.task_label = task_label
        assert task_label in self.SUPPORTED_LABELS
        self.preload()
        super().__init__(seed)

    def loadFile(self, fpath):
        real_fpath = os.path.join(SENTEVAL_DATA_BASE, fpath)
        positives = []
        negatives = []
        with io.open(real_fpath, 'r', encoding='latin-1') as f:
            for line in f:
                target, sample = line.strip().split(':', 1)
                sample = sample.split(' ', 1)[1].lower().split()
                assert target in self.SUPPORTED_LABELS

                if target == self.task_label:
                    positives.append(sample)
                else:
                    negatives.append(sample)
        return positives, negatives

    def preload(self):
        self.train_pos, self.train_neg = self.loadFile('TREC/train_5500.label')
        self.test_pos, self.test_neg = self.loadFile('TREC/TREC_10.label')


    def load_positives(self):
        return self.train_pos + self.test_pos

    def load_negatives(self):
        return self.train_neg + self.test_neg

    def name(self):
        return self.__class__.__name__+'-'+self.task_label


class DebugDataset(CRDataset):
    def __init__(self, seed=1111):
        super().__init__(seed)

        self.positives = self.positives[:100]
        self.negatives = self.negatives[:100]
        self.samples = self.positives + self.negatives
        self.labels = np.array([1] * len(self.positives) + [0] * len(self.negatives))
        self.n_samples = len(self.samples)
        self.reshufle(seed)


ALL_DATASETS = [CRDataset(), MRDataset(), SUBJDataset(), MPQADataset()]
TREC_DATASETS = [TRECDataset(task_label=task) for task in TRECDataset.SUPPORTED_LABELS]

