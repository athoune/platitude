import struct
from io import BytesIO
from pathlib import Path

import lz4.block


class Archive:
    def __init__(self, path: str):
        self.archive_fd = open(f"{path}/archive", "xb+")

        self.index_archive_file = Path(path) / "archive_block_index"
        self.index_archive_fd = self.index_archive_file.open("xb+")
        self.index_archive_format = "L"

        self.index_data_file = Path(path) / "archive_data_index"
        self.index_data_fd = self.index_data_file.open("xb+")
        self.index_data_format = "II"

        self.buffer = BytesIO()
        self.block_size = 1024**2
        self.n_archive = 0

    def write(self, data: bytes):
        self.index_data_fd.write(
            struct.pack(self.index_data_format, len(data), self.n_archive)
        )
        if self.buffer.tell() + len(data) > self.block_size:
            self._write_buffer()
        else:
            self.buffer.write(data)

    def _write_buffer(self):
        self.buffer.seek(0)
        compressed_data = lz4.block.compress(self.buffer.read())
        self.index_archive_fd.write(
            struct.pack(self.index_archive_format, len(compressed_data))
        )
        self.archive_fd.write(compressed_data)
        self.buffer.truncate()
        self.n_archive += 1

    def flush(self):
        self.archive_fd.flush()
        self.index_archive_fd.flush()
        self.index_data_fd.flush()

    def close(self):
        self.flush()
        if self.buffer.tell() > 0:
            self._write_buffer()
        self.archive_fd.close()
        self.index_archive_fd.close()
        self.index_data_fd.close()

    def __len__(self) -> int:
        return self.index_data_file.lstat().st_size // struct.calcsize(
            self.index_data_format
        )

    def _archive_len(self) -> int:
        return self.index_archive_file.lstat().st_size // struct.calcsize(
            self.index_archive_format
        )
