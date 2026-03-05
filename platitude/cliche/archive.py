from io import BytesIO
from pathlib import Path

import lz4.block

from .blockstore import BlockStore
from .index import Index


class Archive:
    def __init__(self, path: str):
        p = Path(path) / "archive"
        p.mkdir()
        self.store = BlockStore(p, "xb+")

        """
         * n_archive: id of the archive
         * index: position in the archive
         * size: size of the archive
        """
        self.index_data = Index(Path(path) / "archive_data_index", "III")

        self.buffer = BytesIO()
        self.block_size = 1024**2
        self.archive_n = 0
        self.archive_poz = 0

    def write(self, data: bytes):
        self.index_data.append(
            self.archive_n,
            self.archive_poz,
            len(data),
        )
        if self.buffer.tell() + len(data) > self.block_size:
            self._write_buffer()
        else:
            self.buffer.write(data)
            self.archive_poz = self.buffer.tell()

    def _write_buffer(self):
        self.buffer.seek(0)
        block: bytes = self.buffer.read()
        compressed_data: bytes = lz4.block.compress(block)
        self.store.append(compressed_data)
        self.buffer.truncate()
        self.archive_n += 1
        self.archive_poz = 0

    def flush(self):
        if self.buffer.tell() > 0:
            self._write_buffer()
        self.store.flush()
        self.index_data.flush()

    def close(self):
        self.flush()
        self.store.close()
        self.index_data.close()

    def __len__(self) -> int:
        return len(self.index_data)

    def _archive_len(self) -> int:
        return len(self.store)

    def __getitem__(self, n) -> bytes:
        n_archive, index, size = self.index_data[n]
        archive = lz4.block.decompress(self.store[n_archive])
        return archive[index : index + size]
