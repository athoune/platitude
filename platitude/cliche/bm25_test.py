from .bm25 import Bm25, ngrams, Vocabulary
from joblib import Parallel, delayed


def test_index():
    index = Bm25()
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]
    index.index(corpus)
    index.compute(2)

    print("Matrix", index.matrix.shape)
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
    for doc in corpus:
        ngs = ngrams(doc, 2)
        print(list(ngs))


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
