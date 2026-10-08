# Optimal Transport for Elastic Full Waveform Inversion

This repository contains code, input models, and saved numerical results for comparing L2, component-wise optimal transport (COT), and rotation-based optimal transport (ROT) misfits in elastic full waveform inversion. It includes landscape tests, linear-model inversion, and Marmousi inversion with refinement and ablation experiments.

## Repository contents

| Directory | Contents |
| --- | --- |
| `Landscape test/` | Landscape calculation, cached results, and figure generation |
| `Linear model test/` | Input models, forward and inversion programs, saved inversion results, and plotting notebook |
| `Marmousi test/` | Input models, forward and inversion programs, SLURM submission files, saved results and timing logs, and plotting notebook |

Each experiment directory contains its original `environment.yml`. These files record the supplied full environments, but they share the same versions of the core numerical dependencies, including Devito, Dask, NumPy, and SciPy.

It is also worth noting that, the original Marmousi model test is designed to run on Tianhe-2 using SLURM and Dask-Jobqueue. Running the inversion on a local CPU system requires adapting the scheduler configuration, resource settings, and file paths.

Synthetic observation data are not included and can be regenerated using the provided forward-modeling scripts.