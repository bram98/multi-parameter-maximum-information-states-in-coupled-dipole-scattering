from pathlib import Path

import dill


def saveobj(obj, name, folder):
    with open(Path(folder) / Path(name), "wb") as file:
        dill.dump(obj, file, dill.HIGHEST_PROTOCOL)


def loadobj(name, folder):
    with open(Path(folder) / Path(name), "rb") as file:
        obj = dill.load(file)
    return obj
