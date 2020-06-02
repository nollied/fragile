from multiprocessing.pool import Pool

import numpy
import xxhash

from fragile.backend.backend import Backend
from fragile.backend.fragile_tensor import tensor


class Hasher:

    _true_hash = bool(Backend.use_true_hash())

    def __init__(self, seed: int = 0):
        self._seed = seed
        if self._true_hash:
            self.pool = Pool()

    @property
    def uses_true_hash(self) -> bool:
        return self._true_hash

    @staticmethod
    def hash_numpy(x: numpy.ndarray) -> int:
        """Return a value that uniquely identifies a numpy array."""
        x = x.astype("|S576") if x.dtype == "O" else x
        return xxhash.xxh64_hexdigest(x.tobytes())

    @staticmethod
    def hash_torch(x):
        bytes = tensor.to_numpy(x).tobytes()
        return xxhash.xxh32_intdigest(bytes)

    @classmethod
    def true_hash_tensor(cls, x):
        funcs = {
            "numpy": cls.hash_numpy,
            "torch": cls.hash_torch,
        }
        return Backend.execute(x, funcs)

    def get_one_id(self):
        self._seed += 1
        return self._seed

    def get_array_of_ids(self, n: int):
        ids = numpy.arange(n) + self._seed + 1
        self._seed += n + 1
        return tensor.as_tensor(ids)

    def hash_tensor(self, x):
        if self._true_hash:
            return self.true_hash_tensor(x)
        return 0

    def hash_walkers(self, x):
        if self._true_hash:
            # hashes = self.pool.map(self.true_hash_tensor, x)
            hashes = [self.true_hash_tensor(x_i) for x_i in x]
            return tensor.as_tensor(hashes)
        return self.get_array_of_ids(x.shape[0])

    def hash_state(self, state):
        if self._true_hash:
            _hash = hash(
                tuple(
                    [
                        self.hash_tensor(x) if k in state._tensor_names else hash(x)
                        for k, x in state.items()
                    ]
                )
            )
            return _hash
        return 0


hasher = Hasher()
