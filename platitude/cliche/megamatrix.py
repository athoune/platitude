import numpy as np
from pathlib import Path


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
        print("row", row, row.itemsize)
        print("poz", self.columns_fd.tell())
        np.uint32(self.matrix_fd.tell()).tofile(self.columns_fd)
        print("length", self.columns_fd.tell(), row.shape[0])
        np.uint32(row.shape[0]).tofile(self.columns_fd)
        row.tofile(self.matrix_fd)
        print("matrix", self.matrix_fd.tell())

    def close(self):
        self.matrix_fd.close()
        self.columns_fd.close()

    def read(self):
        np.fromfile(self.columns_fd, dtype=np.uint32)
        np.fromfile(self.matrix_fd, dtype=self.dtype)

    def _columns(self, key) -> tuple[int, int]:
        poz = key * 8
        tl = np.fromfile(
            self.columns_fd,
            dtype=np.uint32,
            offset=poz - self.columns_fd.tell(),
            count=2,
        )
        return tl[0], tl[1]

    def __getitem__(self, key) -> np.ndarray | int | float | np.unsignedinteger:
        if isinstance(key, int):
            tell, length = self._columns(key)
            self.matrix_fd.seek(0)
            return np.fromfile(
                self.matrix_fd, dtype=self.dtype, offset=tell, count=length
            )
        elif isinstance(key, tuple):
            assert len(key) == 2
            poz = key[0] * 8
            self.columns_fd.seek(0)
            tell = np.fromfile(self.columns_fd, dtype=np.uint32, offset=poz, count=1)[0]
            length = np.fromfile(self.columns_fd, dtype=np.uint32, count=1)[0]
            if key[1] >= length:
                return self.dtype(0)
            self.matrix_fd.seek(0)
            return np.fromfile(
                self.matrix_fd,
                dtype=self.dtype,
                offset=tell + key[1] * np.dtype(self.dtype).itemsize,
                count=1,
            )[0]

        else:
            raise NotImplementedError()

    def column(self, c):
        size: int = int(self.columns_path.lstat().st_size / 8)
        return np.array([self[row, c] for row in range(size)], dtype=self.dtype)
