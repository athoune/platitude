from io import BytesIO
from pathlib import Path

import lz4.block

from .blockstore import BlockStore
from .index import Index


class Archive:
    def __init__(self, path: Path, prefix="archive_", mode="xb+"):
        self.store = BlockStore(path, prefix=prefix, mode=mode)

        """
         * n_archive: id of the archive
         * index: position in the archive
         * size: size of the archive
        """
        self.index_data = Index(path / f"{prefix}block_index", "III", mode=mode)

        self.buffer = BytesIO()
        self.block_size = 1024**2
        self.archive_n = 0
        self.archive_poz = 0
        self.size_compressed = 0
        self.size_data = 0

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
        self.size_data += len(block)
        compressed_data: bytes = lz4.block.compress(block)
        self.size_compressed += len(compressed_data)
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

    def compression_ratio(self) -> float:
        return self.size_compressed / self.size_data
