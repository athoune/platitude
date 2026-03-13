from pathlib import Path

import numpy as np

from .archive import Archive


class Matrix:
    """
    A huge matrix, with an infinite number of columns.
    Rows are append one by one.
    Getting row is cheap, but you can get column too.
    You can get a specific point.
    """

    def __init__(self, path: Path, dtype=np.uint32, mode="xb+", prefix=""):
        self.data = Archive(path, prefix=prefix, mode=mode)
        self.dtype = dtype

    def append_row(self, row: np.ndarray):
        assert len(row.shape) == 1, "One dimension array, not a matrix."
        assert row.dtype == self.dtype
        self.data.write(row.tobytes())

    def close(self):
        self.data.close()

    def flush(self):
        self.data.flush()

    def _columns(self, key) -> tuple[int, int]:
        poz = key * 16
        tl = np.fromfile(
            self.columns_fd,
            dtype=np.uint64,
            offset=poz - self.columns_fd.tell(),
            count=2,
        )
        return tl[0], tl[1]

    def __getitem__(self, key):
        assert isinstance(key, tuple)
        assert len(key) == 2
        return Infinite_row(self.row(key[0]))[key[1]]

    def row(self, r) -> np.ndarray:
        return np.frombuffer(self.data[r], dtype=self.dtype)

    def column(self, c):
        return np.array(
            [
                self[
                    (
                        row,
                        c,
                    )
                ]
                for row in range(len(self))
            ],
            dtype=self.dtype,
        )

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return MatrixIterator(self)


class Infinite_row:
    def __init__(self, row: np.ndarray):
        self.row = row

    def __getitem__(self, key: int):
        if key >= len(self.row):
            return 0
        return self.row[key]


class MatrixIterator:
    def __init__(self, matrix: Matrix):
        self.matrix = matrix
        self.lines = iter(range(len(matrix)))

    def __next__(self):
        line = next(self.lines)
        return self.matrix.row(line)
