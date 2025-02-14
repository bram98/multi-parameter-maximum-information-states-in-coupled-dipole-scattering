# Multi-Parameter Maximum Information States in Coupled Dipole Scattering

## Project Overview
This project explores the application of maximum information states (MIS) in complex scattering environments using the coupled dipole method (CDM). The goal is to:
1. Construct an S-matrix using the CDM framework as described in the paper "Influence of the Local Scattering Environment on the Localization Precision of Single Particles".
2. Optimize the Fisher Information to identify MIS via eigenvalue decomposition.
3. Extend the Fisher Information approach to multi-parameter estimation.
4. Explore other optimal states via auto-grad optimization that are ideal for balanced multi-parameter estimation, or that become insensitive to nuisance parameters.
5. (Future idea) Calculate and visualize Fisher Information flow.

## Related Work
- "Influence of the Local Scattering Environment on the Localization Precision of Single Particles"
  DOI: [10.1103/PhysRevLett.124.133903](https://doi.org/10.1103/PhysRevLett.124.133903)

- "Maximum Information States for Coherent Scattering Measurements"
  DOI: [10.1038/s41567-020-01137-4](https://doi.org/10.1038/s41567-020-01137-4)

- "Continuity equation for the flow of Fisher information in wave scattering"
  DOI: [10.1038/s41567-024-02519-8](https://doi.org/10.1038/s41567-024-02519-8)

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