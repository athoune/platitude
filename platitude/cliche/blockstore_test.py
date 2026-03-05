from pathlib import Path
from tempfile import TemporaryDirectory

from .blockstore import BlockStore


def test_blockstore():
    tmp = TemporaryDirectory()
    store = BlockStore(Path(tmp.name), "ab+")
    size = 8
    for i in range(42):
        block = bytes(
            [
                i,
            ]
            * size
        )
        print("block", i, len(block), block)
        store.append(block)
    store.flush()

    assert len(store) == 42
    block = store[12]
    print("block:", block)
    assert len(block) == size
    assert block[0] == 12
    assert block == bytes(
        [
            12,
        ]
        * size
    )

    i = 0
    for block in store:
        i += 1
    assert i == 42

    store.close()

    tmp.cleanup()
