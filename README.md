# Multi-Parameter Maximum Information States in Coupled Dipole Scattering

## Project Overview
This project explores the application of maximum information states (MIS) in complex scattering environments using the coupled dipole method (CDM). The goal is to:
1. Construct an S-matrix using the CDM framework as described in the paper "Influence of the Local Scattering Environment on the Localization Precision of Single Particles".
2. Optimize the Fisher Information to identify MIS via eigenvalue decomposition.
3. Extend the Fisher Information approach to multi-parameter estimation, specifically focusing on a 3-parameter system involving translations (e.g., x, z) and rotation.
4. Explore various optimal input states via auto-grad optimization, targeting objectives such as maximizing the Fisher Information Matrix determinant (det(J)), minimizing Tr(J⁻¹), or minimizing the spectral radius of J⁻¹. This includes developing strategies to handle nuisance parameters, for instance, by optimizing objectives like `det(J_interest|nuisance)`, `Tr((J_interest|nuisance)⁻¹)`, or the smallest eigenvalue of `J_interest|nuisance`.
5. Perform comprehensive comparisons of these optimal states by analyzing Cramer-Rao Lower Bounds (CRLBs)
6. Study nuisance parameters and visualizing Pareto frontiers to understand performance trade-offs in multi-parameter estimation.

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

## How to use
After installing, go to the main folder. Run the `make_fisher_matrix.py` file. The files are designed to be run cell-by-cell. Sometimes I use `sys.exit()` after the import cell for easy reloading in Spyder, but they are commented out usually. In `make_fisher_matrix` the simulation is setup, the different lossfunctions are defined and then optimized to maximize Fisher information. The result is a `LossfunctionsDict` which stores every different lossfunction as well as its optimal Fisher matrix and wavefront. The results are saved to disk in the `data/` folder.

Next, go to the file `generate_plots.py`. The results from `make_fisher_matrix.py` are loaded from disk and the different cells are used to generate all plots used in the paper or presentations. The plots are saved in `figures/`. The only exception is for the 'Bias' plot in the appendix, which is generated inside the `nuisance_aware_estimation.py` because of time constraints.

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