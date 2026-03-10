from collections import Counter, defaultdict
import re
from pathlib import Path
from typing import Iterable, Self

import numpy as np
from tqdm import tqdm

from .megamatrix import Matrix


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

    def __getitem__(self, n):
        return list(self.vocab.keys())[n]

    def dump(self, writer):
        for i, (token, rank) in enumerate(self.vocab.items()):
            assert (
                i == rank
            ), f"Vocabulary mismatch, {i}th element should not have rank {rank}"
            writer.write(token)
            writer.write("\n")
        writer.flush()

    @classmethod
    def loadVocabulary(cls, reader) -> Self:
        v = Vocabulary()
        for i, line in enumerate(reader.readlines()):
            v.vocab[line[:-1]] = i
        return v


class Bm25:
    def __init__(self, path: Path, ngram_start: int = 3, ngram_end: int = 4):
        self.vocabulary = Vocabulary()
        print("path", f"{path}/data")
        self.documents = Matrix(path, prefix="documents", dtype=np.uint32)
        self.ngram_start = ngram_start
        self.ngram_end = ngram_end
        self.token_counter = defaultdict[int, int](int)

    def index(self, corpus: Iterable[str]) -> None:
        assert not isinstance(corpus, str)
        for doc in corpus:
            tokens = ngrams(doc, self.ngram_start, self.ngram_end)
            ids = self.vocabulary.to_ids(tokens)
            for i in set(ids):
                self.token_counter[i] += 1
            row = np.zeros(self.vocabulary.idx, dtype=np.uint32)
            for token in ids:
                row[token] += 1
            self.documents.append_row(np.trim_zeros(row, "b"))
        self.documents.flush()

    def flush(self):
        self.documents.flush()

    def compute(self, n_jobs: int = -1):
        # self.matrix, self.mask = self.shrinked_matrix(0.1)
        self._IDF = np.zeros(len(self.vocabulary), dtype=np.float32)
        self._avgdl = np.float32(0)

    def n(self, q: int) -> int:
        return self.token_counter[q]

    def f(self, q: int, D: int) -> np.int32:
        return self.documents[D][q]

    def len_D(self, D: int) -> int:
        return self.documents[D].sum()

    def N(self) -> int:
        return len(self.documents)

    def IDF(self, q: int) -> np.float32:
        nq = self.n(q)
        return np.log((self.N() - nq + 0.5) / (nq + 0.5) + 1)

    def IDF_all(self, data: np.ndarray) -> np.ndarray:
        N = data.shape[0]
        return np.array([np.log((N - nq + 0.5) / (nq + 0.5) + 1) for nq in data])

    def avgdl(self) -> np.float32:
        if self._avgdl == 0:
            self._avgdl = sum(
                len(doc.nonzero()[0]) for doc in tqdm(self.documents, desc="agdl")
            ) / len(self.documents)
        return self._avgdl

    def score(self, q: int, D: int):
        if self.documents[D][q] == 0:
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
            desc="Vocab stats",
        ):
            yield k, self.token_counter[v] * self.IDF(v)
            # / len(self.matrix[:, v].nonzero()[0])

    def vocab_stats_all(self, threshold: float = 0):
        self.IDF_all(self.documents.sum[0])

    def top(self, n=10):
        return sorted(self.vocab_stats(), key=lambda x: x[1], reverse=True)[:n]


RE_SENTENCES = re.compile(r"[.,!?]+| - ")


def ngrams(doc: str, start: int = 1, end: int = 1):
    for sentence in RE_SENTENCES.split(doc):
        words = sentence.lower().strip("\n ").split()
        for j in range(start, end + 1):
            for i in range(len(words) - j + 1):
                yield " ".join(words[i : i + j])
