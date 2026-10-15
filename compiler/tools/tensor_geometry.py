"""Logical tensor views. No placement, cycle, or instruction decisions."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Value:
    node: int
    output: int
    shape: tuple
    dtype: str
    indices: np.ndarray
    owners: np.ndarray | None = None

    @property
    def size(self): return int(np.prod(self.shape))
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple): shape = shape[0]
        geometry = self.indices.reshape(self.shape).reshape(*shape)
        owners = None if self.owners is None else self.owners.reshape(self.shape).reshape(*shape).reshape(-1)
        return Value(self.node, self.output, geometry.shape, self.dtype, geometry.reshape(-1), owners)
    def __getitem__(self, item):
        geometry = self.indices.reshape(self.shape)[item]
        owners = None if self.owners is None else self.owners.reshape(self.shape)[item].reshape(-1)
        return Value(self.node, self.output, geometry.shape, self.dtype, geometry.reshape(-1), owners)
    def repeat(self, count, axis):
        geometry = np.repeat(self.indices.reshape(self.shape), count, axis=axis)
        owners = None if self.owners is None else np.repeat(self.owners.reshape(self.shape), count, axis=axis).reshape(-1)
        return Value(self.node, self.output, geometry.shape, self.dtype, geometry.reshape(-1), owners)
    @property
    def T(self):
        geometry = self.indices.reshape(self.shape).T
        owners = None if self.owners is None else self.owners.reshape(self.shape).T.reshape(-1)
        return Value(self.node, self.output, geometry.shape, self.dtype, geometry.reshape(-1), owners)

    def segments(self):
        if self.owners is None:
            yield (self.node, self.output), self.indices, np.arange(self.size)
        else:
            for owner in np.unique(self.owners):
                positions = np.flatnonzero(self.owners == owner)
                yield (int(owner) // 256, int(owner) % 256), self.indices[positions], positions

    @staticmethod
    def concatenate(values, axis=0):
        indices = np.concatenate([v.indices.reshape(v.shape) for v in values], axis=axis)
        owners = np.concatenate([(np.full(v.shape, v.node * 256 + v.output, np.int64)
            if v.owners is None else v.owners.reshape(v.shape)) for v in values], axis=axis)
        return Value(-1, 0, indices.shape, values[0].dtype, indices.reshape(-1), owners.reshape(-1))
