# storable.py
from pathlib import Path
from typing import Type, TypeAlias, TypeVar

import dill

T = TypeVar("T", bound="Storable")


class Storable:
    def save(self, name: str, folder: str = ".") -> None:
        path = Path(folder) / name
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls: Type[T], name: str, folder: str = ".") -> T:
        path = Path(folder) / name
        with open(path, "rb") as f:
            obj = dill.load(f)
        return obj
