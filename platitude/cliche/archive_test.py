from .archive import Archive
from tempfile import TemporaryDirectory


def test_archive():
    tmp = TemporaryDirectory()
    a = Archive(tmp.name)
    a.block_size = 100
    a.write(
        bytes(
            [
                0,
            ]
            * 42
        )
    )
    a.write(
        bytes(
            [
                1,
            ]
            * 27
        )
    )
    a.write(
        bytes(
            [
                2,
            ]
            * 32
        )
    )
    a.close()
    assert len(a) == 3
    assert a._archive_len() == 2

    tmp.cleanup()
