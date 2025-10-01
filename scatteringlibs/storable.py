# storable.py
from pathlib import Path
from typing import TYPE_CHECKING, Type, TypeVar

import dill

"""
This typevar magic makes the return type of load the same as the class that 
inherits from Storable
"""
# if "T" not in globals():
#     T = TypeVar("T", bound="Storable")
# T = TypeVar("T", bound="Storable")
if TYPE_CHECKING:
    # Only mypy / your IDE sees this.  At run‐time, TYPE_CHECKING is False,
    # so Python never creates a real TypeVar object.
    from typing import TypeVar

    T = TypeVar("T", bound="Storable")
else:
    # At run‐time, “T” is just an alias for Storable itself (or a dummy).
    # You could also write “T = None” or “T = object” here—anything that
    # makes the annotation syntactically valid but isn’t a real TypeVar.
    T = None  # <--- no actual TypeVar at run‐time


class Storable:
    def save(self, name: str = None, folder: str = ".") -> None:

        folder_path = Path(folder)

        if not folder_path.exists():
            print(f"Creating folder {folder_path}")
            folder_path.mkdir(exist_ok=False, parents=True)

        if name is None:
            name = self.__class__.__name__
        path = folder_path / name

        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls: Type[T], name: str = None, folder: str = ".") -> T:
        if name is None:
            name = cls.__name__

        path = Path(folder) / name
        with open(path, "rb") as f:
            obj = dill.load(f)
        return obj
