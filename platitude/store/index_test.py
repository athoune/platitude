from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import raises

from .index import Index


def test_index():
    tmp = TemporaryDirectory()
    index = Index(Path(tmp.name) / "idx", "II")

    index.append(42, 12)
    index.append(48, 7)
    with raises(Exception) as e:
        index.append(4, 5, 6)
        assert e.args[0].find("values, not") != -1
    index.flush()

    assert len(index) == 2
    assert index[1] == (48, 7)
    assert index.tell() == 4 * 2 * 2  # 2 rows of 2 32bits

    index.close()
    tmp.cleanup()
