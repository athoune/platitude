from pathlib import Path

import numpy as np


class Matrix:
    """
    A huge matrix, with an infinite number of columns.
    Rows are append one by one.
    Getting row is cheap, but you can get column too.
    You can get a specific point.
    """

    def __init__(self, path: str, dtype=np.uint32, mode="xb+"):
        self.columns_path = Path(f"{path}_columns")
        self.matrix_fd = open(path, mode=mode)
        self.columns_fd = open(f"{path}_columns", mode=mode)
        self.dtype = dtype

    def append_row(self, row: np.ndarray):
        assert len(row.shape) == 1, "One dimension array, not a matrix."
        assert row.dtype == self.dtype
        np.uint64(self.matrix_fd.tell()).tofile(self.columns_fd)
        np.uint64(row.shape[0]).tofile(self.columns_fd)
        row.tofile(self.matrix_fd)

    def close(self):
        self.matrix_fd.close()
        self.columns_fd.close()

    def flush(self):
        self.matrix_fd.flush()
        self.columns_fd.flush()

    def read(self):
        np.fromfile(self.columns_fd, dtype=np.uint32)
        np.fromfile(self.matrix_fd, dtype=self.dtype)

    def _columns(self, key) -> tuple[int, int]:
        poz = key * 16
        tl = np.fromfile(
            self.columns_fd,
            dtype=np.uint64,
            offset=poz - self.columns_fd.tell(),
            count=2,
        )
        return tl[0], tl[1]

    def __getitem__(self, key) -> np.ndarray:
        assert isinstance(key, tuple)
        assert len(key) == 2
        poz: int = key[0] * 16
        tell: np.uint32 = np.fromfile(
            self.columns_fd,
            dtype=np.uint64,
            offset=poz - self.columns_fd.tell(),
            count=1,
        )[0]
        length = np.fromfile(self.columns_fd, dtype=np.uint64, count=1)[0]
        if key[1] >= length:
            return self.dtype(0)
        self.matrix_fd.seek(0)
        return np.fromfile(
            self.matrix_fd,
            dtype=self.dtype,
            offset=tell + key[1] * np.dtype(self.dtype).itemsize,
            count=1,
        )[0]

    def row(self, r) -> np.ndarray:
        tell, length = self._columns(r)
        return np.fromfile(
            self.matrix_fd,
            dtype=self.dtype,
            offset=int(tell) - self.matrix_fd.tell(),
            count=length,
        )

    def column(self, c):
        return np.array([self[row, c] for row in range(len(self))], dtype=self.dtype)

    def __len__(self):
        return int(self.columns_path.lstat().st_size / 16)

    def __iter__(self):
        return MatrixIterator(self)


class MatrixIterator:
    def __init__(self, matrix: Matrix):
        self.matrix = matrix
        self.lines = iter(range(len(matrix)))
        self.matrix.matrix_fd.seek(0)
        self.matrix.columns_fd.seek(0)

    def __next__(self):
        line = next(self.lines)
        return self.matrix.row(line)
