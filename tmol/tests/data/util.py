import pickle
import attr

from collections.abc import Mapping


@attr.s
class LazyFileMapping(Mapping):
    file_mapping = attr.ib()
    norm = attr.ib(default=lambda i: i)

    @classmethod
    def from_list(cls, file_list, norm=lambda i: i):
        file_mapping = {norm(f): f for f in file_list}

        return cls(file_mapping=file_mapping, norm=norm)

    def __len__(self):
        return len(self.file_mapping)

    def __iter__(self):
        return iter(self.file_mapping)


class LazyContentsMapping(LazyFileMapping):
    def __getitem__(self, k):
        with open(self.file_mapping[self.norm(k)], "r") as f:
            return f.read()
