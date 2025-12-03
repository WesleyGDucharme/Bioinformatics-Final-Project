# Bioinformatics-Final-Project
Final project for CSC427 at Uvic. For this project we are to find a bioinformatics paper and recreate / reimplement their code and experiments to then compare our results with the original authors. This is done to get experience testing the reproducibility of experiments and performing said experiments. 

Kameris reimplementation:

Covering experiment steps from the paper "An open-source k-mer based machine learning tool for fast and accurate subtyping of HIV-1 genomes"
By authors Stephen Solis-Reyes, Mariano Avino, Art Poon, and Lila Kari.


## Quick setup

- Requires Python 3.10+.
- Install dependencies (NumPy, SciPy, scikit-learn, PyYAML, matplotlib if you want plots):
  ```bash
  python3 -m pip install -e .
  # or set PYTHONPATH=src for ad-hoc runs without installing
  ```

## Repository layout

```
.
├─ src/                          # kameris_reimp package (k-mer, preprocessing, models, datasets)
├─ scripts/                      # runner and helper scripts (CV, benchmark, label builders, plots)
├─ experiments/                  # YAML experiment specs
├─ data/                         # datasets (labels, manifests, per-file FASTA directories)
├─ results/                      # JSON outputs (e.g., main_experiments_resutls_k6, main_experiments_results_k=5_V1)
├─ figures/                      # generated figures
├─ tests/                        # pytest unit tests
├─ report.md                     # project write-up
├─ test.md                       # scratch notes
├─ CITATION.cff
└─ pyproject.toml
```

Key scripts:
- `scripts/run_experiment.py`, `scripts/run_from_yaml.py` — main experiment runners (set `PYTHONPATH=src`).
- `scripts/run_pol2010_benchmark.py` — train on Web pol2010, test on benchmark pol fragments.
- `scripts/natural_vs_synthetic.py` — natural vs synthetic pol classification.
- `scripts/make_labels_from_json.py`, `scripts/make_flu_labels.py` — label builders for datasets.
- `scripts/modmap_plot.py` and `scripts/make_modmap_*` — mapping/plotting helpers for accession IDs and MoDMaps.


## Data 

- Please refer to the source paper referenced in CITATION.cff and report.md for instructions on how to acquire the data files used for this reimplementation.


## Key scripts and example usages

This project provides several helper scripts in `scripts/`. Prefix commands with `PYTHONPATH=src` (or install the package) so `kameris_reimp` imports resolve. Below are common tasks and example commands.

1) Run a YAML experiment

The experiments are specified as YAML files under `experiments/`. To run an experiment and write outputs to `results/` use:

```bash
PYTHONPATH=src python3 scripts/run_from_yaml.py experiments/hiv1_lanl_whole.yml
```

This runs the configured pipeline (feature extraction, model training, CV) and saves metrics, timings and logs into the chosen output directory.

Example commands assume you’ve installed deps and either run from a virtual env with pip install -e . or prefix with PYTHONPATH=src (as shown).

2) Run the runner with a single-spec programmatic entrypoint

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset dengue_whole \
  --model linear_svm \
  --k 6 \
  --cv-splits 10 \
  --seed 42 \
  --out results/dengue_whole__linear_svm__k6.json
```

3) Create labels files from existing JSON manifests

If you have dataset JSON manifests (see `data/*/*.json`), use the helper to build `labels.tsv`:

```bash
PYTHONPATH=src python3 scripts/make_labels_from_json.py --manifest data/flu_a/metadata.json --out data/flu_a/labels.tsv
```

4) Influenza-specific label generator

```bash
PYTHONPATH=src python3 scripts/make_flu_labels.py --fasta-dir data/flu_a/fasta/ --out data/flu_a/labels.tsv
```

5) Produce a summary of results

```bash
PYTHONPATH=src python3 scripts/summarize_results.py --results-dir results/ --out figures/summary.csv
```

```bash
python3 scripts/make_labels_from_json.py --help
```


## Tests

Run unit tests with pytest:

```bash
pytest -q
```

The tests in `tests/` cover k-mer extraction, preprocessing pipelines and dataset loading utilities. 


## Citation

If you use this work, please cite the original paper (see `CITATION.cff`).

---
