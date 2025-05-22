# <project_root>/setup.py
from setuptools import find_packages, setup

setup(
    name="scatteringlibs",
    version="0.1.0",
    packages=find_packages(include=["scatteringlibs", "scatteringlibs.*"]),
    # find_packages will automatically discover the 'scatteringlibs' package
    # in the current directory. 'include' ensures it and any future submodules are picked up.
    description="Custom libraries for scattering simulations",  # From your old setup.py
    author="Jacob Seifert, Bram Verreussel",  # From your old setup.py
    license="MIT",
    python_requires=">=3.10",
)
