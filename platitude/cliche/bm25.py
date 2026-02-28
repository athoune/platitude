import re
from typing import Generator, Iterable

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


class Vocabulary:
    def __init__(self):
        self.idx = 0
        self.vocab = dict[str, int]()

    def to_ids(self, tokens: Iterable[str]) -> list[int]:
        ids: list[int] = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.idx
                self.idx += 1
            ids.append(self.vocab[token])
        return ids

    def __len__(self) -> int:
        return len(self.vocab)

    def toarray(self) -> np.ndarray:
        return np.array(list(self.vocab.keys()), str)


class Bm25:
    def __init__(self, ngram_size: int = 3):
        self.vocabulary = Vocabulary()
        self.documents: list[np.ndarray] = []
        self.ngram_size = ngram_size

    def index(self, corpus: Iterable[str]) -> None:
        assert not isinstance(corpus, str)
        for doc in corpus:
            ids = self.vocabulary.to_ids(ngrams(doc, self.ngram_size))
            self.documents.append(np.array(ids, dtype=np.int32))

    def compute_matrix(self, n_jobs=-1) -> Generator[np.ndarray, None, None]:
        parallel = Parallel(n_jobs=n_jobs, return_as="generator")
        for line in tqdm(
            parallel(delayed(self._compute_matrix)(doc) for doc in self.documents),
            total=len(self.documents),
            desc="Compute docs",
        ):
            yield line

    def _compute_matrix(self, document: np.ndarray) -> np.ndarray:
        line = np.zeros(len(self.vocabulary), dtype=np.int32)
        for token in document:
            line[token] += 1
        return line

    def compute(self, n_jobs: int = -1):
        self.matrix = np.zeros((len(self.documents), len(self.vocabulary)))
        for i, line in enumerate(self.compute_matrix(n_jobs)):
            self.matrix[i] = line
        self._nq = self.matrix.sum(axis=0)
        self._IDF = np.zeros(len(self.vocabulary), dtype=np.float32)
        self._avgdl = np.float32(0)

    def n(self, q: int) -> int:
        return self._nq[q]
        # return self.matrix[:, q].sum()

    def f(self, q: int, D: int) -> np.int32:
        return self.matrix[D][q]

    def len_D(self, D: int) -> int:
        return self.matrix[D].sum()

    def N(self) -> int:
        return len(self.matrix)

    def IDF(self, q: int) -> np.float32:
        if self._IDF[q] == 0:
            nq = self.n(q)
            self._IDF[q] = np.log((self.N() - nq + 0.5) / (nq + 0.5) + 1)
        return self._IDF[q]

    def avgdl(self) -> np.float32:
        if self._avgdl == 0:
            self._avgdl = sum(
                len(doc.nonzero()[0]) for doc in tqdm(self.matrix, desc="agdl")
            ) / len(self.matrix)
        return self._avgdl

    def score(self, q: int, D: int):
        if self.matrix[D][q] == 0:
            return 0
        fqd = self.f(q, D)
        k1 = 1.5
        b = 0.75
        return (
            self.IDF(q)
            * (fqd * k1 + 1)
            / (fqd + k1 * (1 - b + self.len_D(D) / self.avgdl()))
        )

    def vocab_stats(self):
        for k, v in tqdm(
            self.vocabulary.vocab.items(),
            total=len(self.vocabulary),
            desc="Vocab stats",
        ):
            yield k, self.matrix[:, v].sum() * self.IDF(v)
            # / len(self.matrix[:, v].nonzero()[0])

    def top(self, n=10):
        return sorted(self.vocab_stats(), key=lambda x: x[1], reverse=True)[:n]


RE_SENTENCES = re.compile(r"[.,!?]+| - ")


def ngrams(doc: str, n: int = 1):
    for sentence in RE_SENTENCES.split(doc):
        words = sentence.lower().strip().split()
        for i in range(len(words) - n + 1):
            yield " ".join(words[i : i + n])
