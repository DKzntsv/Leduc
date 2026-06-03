# Leduc Poker — CFR vs Deep Self-Play

Bachelor-thesis codebase (Bocconi BAI). The single Jupyter notebook
`leduc.ipynb` is the source of every number, curve, and table the
written thesis reports. The full specification lives in `[PLAN.md](PLAN.md)` — read
it before changing anything.

**Thesis:** *Approximating Nash Equilibria in Imperfect-Information Games: A
Comparison of CFR and Deep Self-Play on Leduc Poker.*

## Current status

First checkpoint (`PLAN.md` §11): sections **§0 Setup**, **§1 Game & ruler**, and
**§2 CFR / CFR+ baseline** are implemented and verified. On 2 000 CFR iterations
the average policy reaches:


| method | final exploitability (chips / hand) |
| ------ | ----------------------------------- |
| CFR    | ≈ 7 × 10⁻³                          |
| CFR+   | ≈ 9 × 10⁻⁵                          |


CFR+ beats CFR by roughly two orders of magnitude — the expected qualitative
result. The trained average policies are saved under `data/checkpoints/` and are
the reference equilibrium for evaluating every deep method added in §3 – §10.

## Repository layout

```
PLAN.md                       — single source of truth for the thesis experiment
leduc_nash_comparison.ipynb   — the deliverable notebook (run top-to-bottom)
requirements.txt              — pinned Python deps
scripts/build_notebook.py     — regenerates the .ipynb from Python source
                                (the only sane way to diff a notebook in git)
data/
  checkpoints/                — saved tabular CFR strategies + NN weights
  results/                    — per-run CSVs backing every figure
  figures/                    — final PDF (vector, for LaTeX) + PNG previews
```

## Install

OpenSpiel 1.6.x ships wheels for Python **3.11 – 3.13** only; 3.14 is not yet
supported. The notebook is tested against Python 3.13 on Apple Silicon (MPS).

```bash
# 1. Create the venv with a supported interpreter.
python3.13 -m venv .venv

# 2. Install pinned dependencies.
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Run

### From the command line (headless re-execution)

Runs every cell top-to-bottom and writes outputs back into the notebook:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  leduc_nash_comparison.ipynb --ExecutePreprocessor.timeout=900
```

§0 – §2 takes ≈ 3 minutes on an Apple M-series CPU (CFR is pure-Python tabular;
the GPU is unused at this stage).

### Interactively (JupyterLab)

```bash
.venv/bin/jupyter lab leduc_nash_comparison.ipynb
```

Selecting the venv kernel is automatic if you launch `jupyter` from inside
`.venv`. Otherwise register it once:

```bash
.venv/bin/python -m ipykernel install --user --name leduc-venv \
  --display-name "Python 3 (Leduc venv)"
```

## Device selection

The notebook auto-picks the best available PyTorch backend in this order:

1. **MPS** — Apple Silicon (M-series), used here on M5.
2. **CUDA** — NVIDIA GPU if present.
3. **CPU** — fallback. Leduc is tiny enough that CPU is fully sufficient even
  for the deep agents; MPS is a convenience, not a requirement.

The chosen device is printed at the top of §0 and stored in `CONFIG.device`.

## Editing the notebook

Jupyter notebooks diff badly in git. The notebook is therefore generated from
`scripts/build_notebook.py` — edit the Python source and regenerate:

```bash
.venv/bin/python scripts/build_notebook.py
```

If you change cells directly in JupyterLab while iterating, that is fine — just
fold the change back into `scripts/build_notebook.py` before committing so the
two stay in sync.

## Reproducibility

- Global seeds (`random`, `numpy`, `torch`, `torch.mps`) are set from
`CONFIG.seeds[0]` at the start of §0; deep-method seeds will sweep
`CONFIG.seeds = (0, 1, 2, 3, 4)` as required by `PLAN.md` §7.
- Every per-run log is written to `data/results/*.csv` with the schema
`[method, seed, iteration, episodes, wall_clock_s, exploitability]`. Figures
read from those CSVs, so plots can be regenerated without retraining.
- The §2 cells include a save/reload round-trip that verifies the on-disk
average policy reproduces the in-memory exploitability bit-for-bit.

## Contents

- **§3** Deep CFR (`open_spiel.python.pytorch.deep_cfr`)
- **§4** NFSP (`open_spiel.python.pytorch.nfsp`)
- **§5** Policy-gradient self-play — RMPG (`open_spiel.python.pytorch.policy_gradient`)
- **§6** PPO self-play wrapper — *negative baseline* (`open_spiel.python.pytorch.ppo` + custom self-play loop, see `PLAN.md` §5.5)
- **§7** Head-to-head matrix vs the §2 CFR+ equilibrium
- **§8** 3-player Leduc extension
- **§9** Multi-seed robustness band + small hyperparameter sweep
- **§10** Aggregate exploitability-vs-iter / exploitability-vs-wall-clock plots + PDF / CSV exports

