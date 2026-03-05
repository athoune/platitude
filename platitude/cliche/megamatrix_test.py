from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np

from .megamatrix import Matrix


def test_row():
    tmp = TemporaryDirectory()
    matrix = Matrix(Path(tmp.name), prefix="matrix_", dtype=np.int64)
    matrix.append_row(np.array([2, 4, 6, 8], dtype=np.int64))
    matrix.append_row(np.array([7, 8, 9], dtype=np.int64))
    matrix.close()

    matrix2 = Matrix(Path(tmp.name), prefix="matrix_", mode="rb", dtype=np.int64)
    assert not (matrix2.row(0) - np.array([2, 4, 6, 8], dtype=np.int64)).all()
    assert not (matrix2.row(1) - np.array([7, 8, 9], dtype=np.int64)).all()
    assert matrix2[1, 1] == 8
    assert matrix2[0, 0] == 2
    assert matrix2[0, 100] == 0

    data = (np.array([2, 4, 6, 8], dtype=np.int64), np.array([7, 8, 9], dtype=np.int64))
    for i, row in enumerate(matrix2):
        assert not (row - data[i]).all()

    matrix2.close()
    tmp.cleanup()


def test_column():
    tmp = TemporaryDirectory()
    matrix = Matrix(Path(tmp.name), prefix="matrix_", dtype=np.int64)
    matrix.append_row(np.array([2, 4, 6, 8], dtype=np.int64))
    matrix.append_row(np.array([7, 8, 9], dtype=np.int64))
    matrix.close()

    matrix2 = Matrix(Path(tmp.name), prefix="matrix_", mode="rb", dtype=np.int64)

    assert not (matrix2.column(1) - np.array([4, 8], dtype=np.int64)).all()
    assert not (matrix2.column(3) - np.array([8, 0], dtype=np.int64)).all()
    tmp.cleanup()
