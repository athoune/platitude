from pathlib import Path

from .index import Index


class BlockStore:
    def __init__(self, path: Path, mode="xb+"):
        p = Path(path) / "index"
        self.index = Index(p, "LI")
        self.data_fd = open(f"{path}/data", mode=mode)

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
