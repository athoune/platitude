import struct
from pathlib import Path


class Index:
    def __init__(self, path: Path, format: str):
        self.path = path
        self.fd = self.path.open(mode="xb+")
        self.pack = struct.Struct(format)
        self._arity = 0

    def __getitem__(self, n: int) -> tuple:
        self.fd.seek(n * self.pack.size, 0)
        return self.pack.unpack(self.fd.read(self.pack.size))

    def append(self, *values):
        if self._arity == 0:
            self._arity = len(values)
        elif self._arity != len(values):
            raise Exception(f"{self._arity} values, not {len(values)}")

        self.fd.write(self.pack.pack(*values))

    def flush(self):
        self.fd.flush()

    def close(self):
        self.fd.close()

    def tell(self):
        return self.fd.tell()

    def __len__(self) -> int:
        return self.path.lstat().st_size // self.pack.size
