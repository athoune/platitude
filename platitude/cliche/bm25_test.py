from pathlib import Path
from tempfile import TemporaryDirectory

from .bm25 import Bm25, Vocabulary, ngrams


def test_index():
    index = Bm25()
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]
    index.index(corpus)
    assert len(index.documents) == len(corpus)

    index.compute(2)
    assert index.matrix.shape == (len(index.documents), len(index.vocabulary))

    stats = index.vocab_stats()
    print(list(stats))


def test_ngrams():
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
        "don't eat cats. Nor dogs",
    ]
    ngs = ngrams(corpus[0], 2)
    assert list(ngs)[:4] == ["a cat", "cat is", "is a", "a feline"]


def test_vocabulary():
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
        "don't eat cats. Nor dogs",
    ]
    vocabulary = Vocabulary()
    for doc in corpus:
        ids = vocabulary.to_ids(ngrams(doc, 2))
        print(ids)

    arrays = vocabulary.toarray()
    assert len(arrays) == len(vocabulary.vocab)
    for token in arrays:
        assert token in vocabulary.vocab


def test_vocab_stats():
    tmp = TemporaryDirectory()
    index = Bm25(Path(tmp.name), 2, 3)
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
        "I want to eat a cat",
    ]
    index.index(corpus)
    index.flush()
    assert len(index.documents) == len(corpus)

    for k, v in index.vocabulary.vocab.items():
        idf = index.IDF(v)
        print(k, idf)

    stats = dict(index.vocab_stats())
    assert stats["a cat"] > 1
    print("top", index.top())
    assert index.top()[0][0] == "a cat"

    tmp.cleanup()


def test_dump_load():
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
        "don't eat cats. Nor dogs",
    ]
    vocabulary = Vocabulary()
    for doc in corpus:
        vocabulary.to_ids(ngrams(doc, 2))
    bird = vocabulary.to_ids(["a bird"])[0]
    tmp = TemporaryDirectory()
    dump = Path(tmp.name) / "vocab.txt"
    vocabulary.dump(dump.open("w"))

    vocabulary2 = Vocabulary.loadVocabulary(dump.open("r"))
    assert "a bird" == vocabulary2[bird]

    tmp.cleanup()
