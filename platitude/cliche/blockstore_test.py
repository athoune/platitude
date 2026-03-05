from .blockstore import Blockstore
from tempfile import TemporaryDirectory


def test_blockstore():
    tmp = TemporaryDirectory()
    store = Blockstore(tmp.name, "ab+")
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
    store.close()

    tmp.cleanup()
