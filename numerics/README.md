# Code and data for "Learning quantum functions using tensor cross interpolation"

Seongwook Shin, Ryan Sweke, Jens Eisert

This directory contains the code, the stored results and the figure files
behind the *Numerical results* section of the manuscript. Each experiment has
its own folder with the code that was run, the data it produced and the
resulting figures.

## Layout

| Path | Content |
|---|---|
| `01_tmk_vs_ftci/` | Artificial functions: comparison of FTCI with the trigonometric monomial kernel (TMK) |
| `02_heisenberg_hamiltonian/` | Hamiltonian dynamics of the Heisenberg chain |
| `03_noisy_oracle/` | Robustness to shot noise |
| `04_global_observable/` | Global observables on shallow circuit states |
| `05_qcnn/` | Quantum convolutional neural networks |
| `06_decaying_rate/` | Artificial functions: random low-bond-dimension targets and decaying coefficients |
| `modules/` | Julia modules shared by the experiments |
| `env/` | Julia environment (`Project.toml`, `Manifest.toml`) |
| `inspect_jls.jl` | Prints the content of the serialized result files |

Inside an experiment folder, `code/` holds notebooks and scripts, `results/`
the data and `logs/` the run configuration where one was written. `figures/`
holds each figure as it appears in the manuscript (PDF) and, for two
experiments, the individual panels as the code wrote them (SVG).

## Figures, code and data

Figures are named by their label in the manuscript.

| Figure | Folder | Code | Data |
|---|---|---|---|
| `fig:Random low chi_t` | 06 | `TCIjulia_general_omega_clean.ipynb`, cell "For plot: Averaged relative error over 5 seeds" | `results/L_chi_results_avg.csv` |
| `fig:tmk-ftci-spread` | 01 | `TMK_vs_FTCI_full_sweep.ipynb`, which runs `tmk_vs_tci_lowbond.jl` and plots with `modules/TMKvsFTCIPlots.jl`. `plot_tmk_vs_tci_plain.py` redraws the figure from the CSV. | `results/tmk_vs_tci_lowbond_full_20260519_003147.csv`, derived `results/threshold_samples_full_20260519_003147.csv` |
| `fig:tmk-ftci-threshold-multil` | 01 | `tmk_vs_tci_lowbond.jl` for the sweep, `plot_tmk_threshold_combined.py` for the figure | `results/tmk_vs_tci_lowbond_extended50k_lambda4_combined_20260519_164108.csv`, derived `results/threshold0p1_samples_extended50k_lambda4_combined_20260519_164108.csv` |
| `fig:Decaying` | 06 | `TCIjulia_general_omega_clean.ipynb`, cells that sweep the decay rate (`kval`) | `results/kval_avg_results.csv` |
| `fig:QCNN` | 05 | `QCNN_with_TCI_simple_TI.ipynb` | `results/QCNN_all_N_20260125_0244.jls`; `results/QCNN_N{4,8,16,32}.jls` are per-size extracts of the same run |
| `fig:Global observable` | 04 | `TCIjulia_multiple_observable.ipynb`, cell "clean version" | `results/global_observable_summary.csv`, `results/global_observable_per_seed.csv` |
| `fig:ham` | 02 | `HeisenbergFTCI_MPS.ipynb`; `resume_table1_N16.jl` continues the size sweep at the largest size | `results/processed/`, `results/raw/`, configuration in `logs/config_20260611_124951.txt` |
| `fig:noisy` | 03 | `NoisyOracle_FTCI.ipynb`; `print_noisy_numbers.jl` prints the numbers quoted in the text | `results/alpha_sweep_N9_chi12.jls`, configuration in `logs/config_noisy_20260701_123300.txt` |

Notes on individual experiments:

- **02** was run with oversampling factor 2 (run 20260611_124951).
- **03** sweeps the oversampling factor over 1, 2, 3 and the number of shots
  over 10² and 10⁴ with three seeds each, and includes noiseless references.
- **04** queries the circuit on the deterministic grid {0, 2π/3, 4π/3} per
  input, for N = 6 to 12 with five random circuits per size and bond
  dimension at most 16. The notebook did not write result files: the two
  CSV files are transcribed from the output stored in the notebook. The
  number of queries is recorded as mean and standard deviation per size, the
  errors per circuit.
- **04, angle convention.** The manuscript writes the input rotation as
  exp(−i n̂ x_k) with n̂ = (X + Y + Z)/√3. The code builds it with Yao's
  `rot(n, θ)`, which is exp(−i n̂ θ/2). The manuscript's input is therefore
  half the angle used in the code, θ_k = 2 x_k, and the manuscript's
  dictionary (1, cos 2x_k, sin 2x_k) is the code's (1, cos θ_k, sin θ_k). The
  grid {0, 2π/3, 4π/3} is the same set of circuits in both conventions,
  because angles are defined up to a full turn. Only the labels of two points
  swap: the code reads out Z, Y, X at θ = 0, 2π/3, 4π/3, the manuscript Z, X,
  Y at x = 0, 2π/3, 4π/3. The results do not depend on the convention.
- The notebooks also contain exploratory cells that are not used in the
  manuscript. Their data files are not included.

## Requirements

- Julia 1.11 (the results were produced with 1.11.6). `env/Manifest.toml`
  pins all package versions:

  ```sh
  julia --project=env -e 'using Pkg; Pkg.instantiate()'
  ```

- Jupyter with the IJulia kernel for the notebooks. IJulia is part of `env`.
- Python 3 for the two plotting scripts in `01_tmk_vs_ftci/code`. They use
  the standard library only.

## Looking at the stored results

`.jls` files are written with Julia's `Serialization` standard library, which
is tied to the Julia version, so read them with Julia 1.11:

```sh
julia --project=env inspect_jls.jl            # every .jls file
julia --project=env inspect_jls.jl 05_qcnn    # one experiment
```

The numbers quoted for the shot-noise experiment:

```sh
julia --project=env 03_noisy_oracle/code/print_noisy_numbers.jl
```

The plotting scripts take the CSV file as argument and write their output
next to it:

```sh
python3 01_tmk_vs_ftci/code/plot_tmk_vs_tci_plain.py 01_tmk_vs_ftci/results/tmk_vs_tci_lowbond_full_20260519_003147.csv
python3 01_tmk_vs_ftci/code/plot_tmk_threshold_combined.py 01_tmk_vs_ftci/results/tmk_vs_tci_lowbond_extended50k_lambda4_combined_20260519_164108.csv
```

## Running the experiments again

The notebooks were run from one flat working directory. They activate the
environment in the current directory with `Pkg.activate(".")`, load the
shared modules with `include("<file>.jl")`, and write below the working
directory. To run one again:

1. Copy `env/Project.toml`, `env/Manifest.toml`, the files in `modules/` and
   the files in the experiment's `code/` folder into one directory.
2. Start Jupyter in that directory and select the Julia 1.11 kernel.
3. Create the output folders named in the `savefig` and `open` calls of the
   cells you run. Some cells do not create them.

| Experiment | Modules loaded |
|---|---|
| 01 | `FunctionTCI.jl`, `TrigonometricMonomialKernel.jl`, `TMKvsFTCIPlots.jl` |
| 02, 03 | `FunctionTCI.jl`, `HeisenbergMPS.jl`, `HeisenbergYao.jl`, `HeisenbergHybrid.jl` |
| 04, 05, 06 | `Modules.jl` |

Random seeds are set in the code.

## Software

The experiments rely on these libraries. Please cite them along with the
manuscript if you build on this code.

- [TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl)
  for tensor cross interpolation: Y. Núñez Fernández *et al.*, "Learning
  tensor networks with tensor cross interpolation: New algorithms and
  libraries", SciPost Phys. 18, 104 (2025).
- [Yao.jl](https://github.com/QuantumBFS/Yao.jl) for statevector simulation:
  X.-Z. Luo, J.-G. Liu, P. Zhang and L. Wang, Quantum 4, 341 (2020).
- [ITensors.jl](https://github.com/ITensor/ITensors.jl) and ITensorMPS.jl for
  matrix-product-state simulation: M. Fishman, S. R. White and
  E. M. Stoudenmire, SciPost Phys. Codebases 4 (2022).

## License

The code in this directory (notebooks, scripts and modules) is released under
the Apache License, Version 2.0. The stored results in the `results/` folders
and the figure files in the `figures/` folders are not covered by the Apache
License.
