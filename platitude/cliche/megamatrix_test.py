from .megamatrix import Matrix
from tempfile import TemporaryDirectory
import numpy as np
import os


def test_row():
    tmp = TemporaryDirectory()
    matrix = Matrix(f"{tmp.name}/matrix_db", dtype=np.int64)
    matrix.append_row(np.array([2, 4, 6, 8], dtype=np.int64))
    matrix.append_row(np.array([7, 8, 9], dtype=np.int64))
    matrix.close()

    matrix2 = Matrix(f"{tmp.name}/matrix_db", mode="rb", dtype=np.int64)
    matrix2.read()
    assert not (matrix2.row(0) - np.array([2, 4, 6, 8], dtype=np.int64)).all()
    assert not (matrix2.row(1) - np.array([7, 8, 9], dtype=np.int64)).all()
    assert matrix2[1, 1] == 8
    assert matrix2[0, 0] == 2
    assert matrix2[0, 100] == 0
    matrix2.close()

    tmp.cleanup()


def test_column():
    tmp = TemporaryDirectory()
    matrix = Matrix(f"{tmp.name}/matrix_db", dtype=np.int64)
    matrix.append_row(np.array([2, 4, 6, 8], dtype=np.int64))
    matrix.append_row(np.array([7, 8, 9], dtype=np.int64))
    matrix.close()

    matrix2 = Matrix(f"{tmp.name}/matrix_db", mode="rb", dtype=np.int64)
    matrix2.read()

    assert not (matrix2.column(1) - np.array([4, 8], dtype=np.int64)).all()
    assert not (matrix2.column(3) - np.array([8, 0], dtype=np.int64)).all()
    tmp.cleanup()
