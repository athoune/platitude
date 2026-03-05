from tempfile import TemporaryDirectory

import lz4.block

from .archive import Archive


def test_archive():
    tmp = TemporaryDirectory()
    a = Archive(tmp.name)
    a.block_size = 100
    assert a.archive_n == 0
    a.write(
        bytes(
            [
                0,
            ]
            * 42
        )
    )
    assert a.archive_n == 0
    a.write(
        bytes(
            [
                1,
            ]
            * 27
        )
    )
    assert a.archive_n == 0
    a.write(
        bytes(
            [
                2,
            ]
            * 32
        )
    )
    assert a.archive_n == 1

    a.flush()

    assert len(a) == 3
    assert a._archive_len() == 2

    for archive in a.store:
        lz4.block.decompress(archive)

    assert a[1] == bytes(
        [
            1,
        ]
        * 27
    )

    a.close()
    tmp.cleanup()
