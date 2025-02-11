# Multi-Parameter Maximum Information States in Coupled Dipole Scattering

## Project Overview
This project explores the application of maximum information states (MIS) in complex scattering environments using the coupled dipole method (CDM). The goal is to:
1. Construct an S-matrix using the CDM framework as described in the paper "Influence of the Local Scattering Environment on the Localization Precision of Single Particles".
2. Optimize the Fisher Information to identify MIS via eigenvalue decomposition.
3. Extend the Fisher Information approach to multi-parameter estimation.
4. (Optional) Compare these states with fields optimized using simulated annealing or other state-of-the-art optimization strategies
5. (Future idea) Calculate and visualize Fisher Information flow.

## Related Work
- "Influence of the Local Scattering Environment on the Localization Precision of Single Particles"  
  DOI: [10.1103/PhysRevLett.124.133903](https://doi.org/10.1103/PhysRevLett.124.133903)

- "Maximum Information States for Coherent Scattering Measurements"  
  DOI: [10.1038/s41567-020-01137-4](https://doi.org/10.1038/s41567-020-01137-4)
  
## Installation
### Prerequisites
Python 3.10 or later is recommended. Use `conda` or `pip` to manage dependencies.
### Steps
1. Clone the repository:
   ```bash
   git clone git@git.science.uu.nl:linx/multi-parameter-maximum-information-states-in-coupled-dipole-scattering.git
   cd multi-parameter-max-info-states
   ```
2. Update the submodules:
```bash
   git submodule update --init
   git submodule update --recursive
```
  After changes in the submodule
```bash
    git submodule update --remote --merge
```
2. Install dependencies with `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   
## Data Folder Usage

The `data/` folder is ignored by Git to prevent large files from being tracked in the repository. This folder is intended for files that:
- Are generated during simulations or analyses.
- Are used locally for testing and development purposes.

## How to create a Python library
An example folder structure for your project (using pip) looks like this:
```
mylibrary/
├─ mylibrary/
│  ├─ __init__.py
│  ├─ mymodule.py
│  ├─ anothermodule.py
├─ README.md
├─ setup.py
```
The content of the `__init__.py` file looks like this:
```python
from . import mymodule
from . import anothermodule
```
The content of `setup.py` could look like the following:
from setuptools import setup, find_packages

```python
from setuptools import setup, find_packages

setup(
    name='mylibrary',
    version='0.1.0',
    packages=find_packages(),  # List all your modules here
    install_requires=['numpy', 'matplotlib', 'scipy'],  # Add dependencies if needed
    description='Collection of modules',
)
```

To install it manually, you can go the root folder (that contains setup.py) and install using pip
```bash
cd path-to-project-folder/mylibrary
pip install .
```
If you want the install to be "editable", you can add the `-e` flag in the command `pip install -e .`. This way, the package is linked instead of installed and any changes in the folder should be reflected immediately.

You can also use a more simple folder structure, but I could not get this to work satisfactorily, so use at your own risk.

```
simplelibrary/
├─ mymodule2.py
├─ anothermodule2.py
├─ README.md
├─ setup.py
```

Now the `setup.py` looks like this:
```python
from setuptools import setup

setup(
    name='simplelibrary',
    version='0.1.0',
    py_modules=['mymodule2', 'anothermodule2'],  # List all your modules here
    install_requires=['numpy'],  # Add dependencies if needed
    description='Collection of modules',
)
```
and to import you can use
```python
import mymodule2, anothermodule2
```


## Contributors
- Bram Verreussel
- Jacob Seifert

## How to Contribute
- Clone this repository and create a branch using your name (e.g., `dev-jacob`).
- Submit pull requests to merge your contributions into the `main` branch.

## License
This project is currently licensed under a **restricted academic license**. It is intended for non-commercial academic research only, and publication or sharing requires prior approval from ASML and Utrecht University.

For detailed licensing terms, see the [LICENSE](./LICENSE) file.

Future changes to this license are possible depending on ASML's decision regarding patents and exclusivity.

## Disclaimer
This software is provided "as-is" without any warranty of any kind, express or implied.
