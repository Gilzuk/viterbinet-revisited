# ViterbiNet Revisited: code and data

Code, results and LaTeX sources for two papers by Gil Zukerman (School of Electrical Engineering, Tel Aviv University):

| Paper | Venue | Source | PDF |
|---|---|---|---|
| *ViterbiNet Revisited: A Matched-Information Classical Baseline for Learned Trellis Detection* | IEEE Communications Letters (to be submitted) | `paper/letter/main.tex` | `paper/letter/main.pdf` |
| *When Does Learning Help Trellis Detection? Matched Baselines, Structure, and Complexity of Model-Based Deep Viterbi Receivers* | IEEE Transactions on Machine Learning in Communications and Networking (to be submitted) | `paper/journal/main.tex` | `paper/journal/main.pdf` |

Every number, table and figure in both papers can be regenerated from the files in this repository. The papers cite the release tag `v1.0-submission`.

## What these papers add to prior work

ViterbiNet [Shlezinger et al., 2020] keeps the Viterbi trellis and learns the branch metric with a small network, adapted online on code-accepted words. Meta-learning was later added to speed up that adaptation [Raviv et al., 2021, 2023]. On Gaussian intersymbol-interference channels, the 2020 paper compares the learned receiver with Viterbi given perfect CSI, CSI with uncertainty, or (under block fading) only the initial CSI; it also compares learned detectors and covers non-Gaussian channels. None of those classical baselines receives the same information as the learned receiver.

| Prior claim or design choice | What these papers find |
|---|---|
| On Gaussian ISI channels the learned metric approaches Viterbi with perfect CSI and beats Viterbi with CSI uncertainty or only initial CSI. | **Matched-information baseline.** LS-Viterbi uses ViterbiNet's pilots and acceptance rule, applied to its own decisions. It has the lowest point-estimate SER of all no-CSI receivers from 0 to 14 dB. On shared draws it is significantly better than ViterbiNet (5 updates) at 1, 6, 7 and 9-12 dB and than VNet-affine at 0, 2-7 and 11 dB, and never significantly worse in any available paired comparison; 13-14 dB is unresolved. It stays within about 3-30% of the perfect-CSI SER, close to a predicted 0.13 dB estimation loss. |
| Online adaptation on code-accepted words, 200 Adam updates per word; meta-learning to adapt faster. | **Adaptation budget.** 200 updates per word overfit; 5 Adam updates give a lower point-estimate SER at every SNR with 40 times fewer updates. The gate of the starting code uses the transmitted bits; under an implementable RS gate LS-Viterbi keeps the lowest point-estimate SER at 7, 10 and 12 dB, though one of twelve paired gate differences (VNet-affine, 7 dB) is significant. Meta-learned adaptation is not tested. |
| A fully connected network learns the state likelihoods (100-50 hidden units in the original; 100-58, 7,002 parameters, here). | **Structure over size.** A 32-parameter affine metric, the exact Gaussian log-likelihood form, matches it. Larger networks are not better. |
| Attention-based architectures, applied here as branch metrics. | Transformer and ViT metrics of the same size are worse at every training budget and cost more to adapt. |
| Cost of learned detection. | Only LS-Viterbi and ViterbiNet with 5 updates fit the measured detection-plus-adaptation time within a 17 ms word on one CPU thread (RS decoding and gating not timed). |
| Higher-order modulation. | For QPSK the per-branch metric needs M^L classes and fails with a realistic pilot. Tying the classes to the channel taps fixes it. |
| Time-varying channels. | The learned receivers' advantage over a last-word LS estimate is memory of the channel statistics, which exponentially weighted LS reproduces. |

In short: in these experiments, on a linear Gaussian ISI channel, the tested learned receivers show no established advantage over a classical receiver given the same information. Learned detectors should be compared against such a receiver.

## Contents

| Path | What it is |
|---|---|
| `Code/` | Simulator and receivers. It covers the COST 2100 ISI channel, shortened Reed-Solomon RS(17,15) coding over GF(2^8) (15 information bytes, 2 parity bytes), the Viterbi detector with perfect, noisy or LS-estimated CSI, ViterbiNet (MLP), VNet-affine, the Transformer metrics and online training (`trainer.py`). `qpsk_sim.py` is the complex QPSK simulator. |
| `Resources/cost2100_channel/` | COST 2100 tap traces. |
| `run_*.py`, `extend_point.py` | Monte Carlo SNR sweeps (see below). |
| `run_mc_sweep_colab.ipynb` | The Colab notebook that ran the ViterbiNet K=200 and Transformer sweeps on a GPU. It is kept as it was run, so it still clones the development repository, `Gilzuk/viterbitransformed`. |
| `experiments/` | Topology, training-budget, latency, QPSK diagnostics, tracker tuning, adaptation-gate and paired-statistics studies. |
| `Results/metrics/` | Every CSV the papers use. `.mc_sweep_checkpoints/` holds the per-repetition SERs used by the paired statistics. |
| `Results/weights/` | Offline-trained weights for every learned receiver in the papers (`*_mcsweep`). |
| `paper/` | LaTeX sources, bibliography, figure generator (`make_figures.py`) and figures. |

## Installing

Python 3.11. Install PyTorch for your platform first (CPU is enough; the papers' results used PyTorch 2.x), then:

```bash
pip install -r requirements.txt
```

## Rebuilding the papers from the stored results

This step needs no simulation and takes about a minute.

```bash
python3 experiments/paired_stats.py      # Results/metrics/paired_ls_vs_learned.csv
python3 paper/make_figures.py            # paper/figures/*.pdf, numbers.tex, paired_table.tex, gate_table.tex
cd paper/letter  && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd ../journal    && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`make_figures.py` applies two bookkeeping corrections to the raw BPSK sweep output, the same for every receiver:
- The stored SER averages pilot words in as zero-error, so the reported SER is the stored value times 125/120.
- Zero-error bounds use the true 14,400 information bits per repetition, not the CSV's nominal count.

`paper/README.md` maps each figure to its CSV and to the script that produced it.

## Re-running the simulations

Each sweep appends rows to `Results/metrics/*.csv` and checkpoints every repetition to `Results/metrics/.mc_sweep_checkpoints/`, so an interrupted run resumes where it stopped.
- Run from the repository root.
- Monte Carlo draws are cached in `Data_Cache/`, which is created on first use. Receivers run from the same cache see identical (bits, noise) for repetitions r < 200; this is what makes the paired comparisons paired.
- Set `OMP_NUM_THREADS=1` when running several sweeps in parallel.

| Result | Command |
|---|---|
| Perfect-CSI Viterbi, BPSK 0-17 dB | `python3 run_mc_sweep.py ClassicViterbi` |
| ViterbiNet K=200, Transformer | `python3 run_mc_sweep.py ViterbiNet` / `Transformer` (the papers' rows came from `run_mc_sweep_colab.ipynb`) |
| LS-Viterbi (no CSI) | `VARIANT_METHOD=Statistical VARIANT_MIN_REPS=100 VARIANT_MAX_BITS=2000000 python3 run_variant_sweep.py ClassicViterbi_LS ClassicViterbi_LS - <snr...>` |
| ViterbiNet, 5 online steps | `python3 run_variant_sweep.py ViterbiNet_on5 ViterbiNet 5 <snr...>` |
| VNet-affine | `python3 run_affine_sweep.py <snr...>` |
| Noisy-CSI Viterbi (25/50/75/100 %) | `python3 run_csi_sweep.py <pct> [start end]` |
| BPSK fast fading | `python3 run_fast_fading.py <snr> <fd> [receivers...]`; tracker tuning with `experiments/ff_tracker_tuning.sh <fd>` |
| QPSK | `python3 run_qpsk_sweep.py`, `python3 run_qpsk_sym_sweep.py`, `python3 run_qpsk_fast_fading.py <snr>`, `python3 experiments/qpsk_diagnostics.py {pilot,trace,window}` |
| Topology and online-step study | `python3 experiments/vnet_study.py`, `python3 experiments/vnet_online_extra.py` |
| Training-budget search | `python3 experiments/mb_search.py` |
| Latency | `python3 experiments/vnet_latency.py`, `python3 experiments/all_latency.py` |
| Oracle vs implementable adaptation gate | `python3 experiments/gate_check.py <snr> [reps]` |
| Paired statistics (LS vs learned; RS vs oracle gate) | `python3 experiments/paired_stats.py` |
| LS estimation-loss check (Gram matrix of coded words) | `python3 experiments/ls_loss_check.py [words] [seed]` |

A full BPSK sweep of one learned receiver takes CPU-days. The stored checkpoints let you reproduce the tables without re-running anything.

## Adaptation gate

The sweeps use the reference ViterbiNet gate (`gate_mode='oracle'` in `Code/trainer.py`). A word adapts the receiver when its decoded SER against the transmitted bits is at most 0.02. Every adaptive receiver uses this gate, LS-Viterbi included.

`gate_mode='rs'` is an implementable alternative:
- It accepts a word when the re-encoded decoded word differs from the hard decisions in at most one RS symbol.
- It always adapts on the re-encoded word.

`experiments/gate_check.py` compares the two gates. The results are in `Results/metrics/gate_check.csv` and in the journal paper's Section "An implementable adaptation gate".

## Citation

If you use this code or these results, please cite:

```bibtex
@unpublished{zukerman2026viterbinetrevisited,
  author = {Gil Zukerman},
  title  = {{ViterbiNet} Revisited: A Matched-Information Classical Baseline for Learned Trellis Detection},
  note   = {Manuscript to be submitted to IEEE Communications Letters},
  year   = {2026},
  url    = {https://github.com/Gilzuk/viterbinet-revisited}}

@unpublished{zukerman2026learningtrellis,
  author = {Gil Zukerman},
  title  = {When Does Learning Help Trellis Detection? Matched Baselines, Structure, and Complexity of Model-Based Deep {Viterbi} Receivers},
  note   = {Manuscript to be submitted to IEEE Transactions on Machine Learning in Communications and Networking},
  year   = {2026},
  url    = {https://github.com/Gilzuk/viterbinet-revisited}}
```

The prior work these papers build on and re-examine:

```bibtex
@article{shlezinger2020viterbinet,
  author  = {Nir Shlezinger and Nariman Farsad and Yonina C. Eldar and Andrea J. Goldsmith},
  title   = {{ViterbiNet}: A Deep Learning Based {Viterbi} Algorithm for Symbol Detection},
  journal = {IEEE Trans. Wireless Commun.},
  volume  = {19}, number = {5}, pages = {3319--3331}, year = {2020},
  doi     = {10.1109/TWC.2020.2972352}}

@inproceedings{raviv2021metaviterbinet,
  author    = {Tomer Raviv and Sangwoo Park and Nir Shlezinger and Osvaldo Simeone and Yonina C. Eldar and Joonhyuk Kang},
  title     = {{Meta-ViterbiNet}: Online Meta-Learned {Viterbi} Equalization for Non-Stationary Channels},
  booktitle = {Proc. IEEE Int. Conf. Commun. Workshops (ICC Workshops)},
  year      = {2021},
  doi       = {10.1109/ICCWorkshops50388.2021.9473693}}

@article{raviv2023online,
  author  = {Tomer Raviv and Sangwoo Park and Osvaldo Simeone and Yonina C. Eldar and Nir Shlezinger},
  title   = {Online Meta-Learning for Hybrid Model-Based Deep Receivers},
  journal = {IEEE Trans. Wireless Commun.},
  volume  = {22}, number = {10}, pages = {6415--6431}, year = {2023},
  doi     = {10.1109/TWC.2023.3241841}}

@article{magee1973adaptive,
  author  = {F. R. Magee and John G. Proakis},
  title   = {Adaptive Maximum-Likelihood Sequence Estimation for Digital Signaling in the Presence of Intersymbol Interference},
  journal = {IEEE Trans. Inf. Theory},
  volume  = {19}, number = {1}, pages = {120--124}, year = {1973}}
```
