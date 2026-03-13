from pathlib import Path

from .index import Index


class BlockStore:
    def __init__(self, path: Path, mode="xb+", prefix=""):
        assert path.is_dir()
        self.path = path
        self.index = Index(self.path / f"{prefix}index", "LI", mode=mode)
        self.data_fd = (self.path / f"{prefix}data").open(mode=mode)

    def append(self, block: bytes):
        self.index.append(self.data_fd.tell(), len(block))
        self.data_fd.write(block)

    def flush(self):
        self.index.flush()
        self.data_fd.flush()

    def close(self):
        self.index.close()
        self.data_fd.close()

    def tell(self) -> int:
        return self.data_fd.tell()

    def __getitem__(self, index) -> bytes:
        poz, size = self.index[index]
        self.data_fd.seek(poz, 0)
        return self.data_fd.read(size)

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        return iterator(self)


class iterator:
    def __init__(self, store):
        self.store = store
        self.n = -1
        self.last = len(store)

    def __next__(self):
        self.n += 1
        if self.n == self.last:
            raise StopIteration
        return self.store[self.n]
