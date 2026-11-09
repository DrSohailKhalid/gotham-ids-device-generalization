# Gotham IDS: Generalization to Unseen Devices

Companion code and experimental outputs for **Beyond Random Splits: Generalization
of IoT Intrusion Detection to Unseen Devices**.

This study evaluates how IoT intrusion detection performance changes when
training and test traffic come from different physical-device identities or
network segments rather than from a random mixture of packets.

## Benchmark and evaluation protocol

- **Dataset:** Gotham Dataset 2025, 78 simulated IoT devices and four segments.
- **Reported working subset:** 404,324 packets, after per-device sampling.
- **Feature regimes:** `F-ALL` (22 raw fields), `F-NOID` (15 fields without
  device addresses/checksums/absolute time), and `F-BEHAV` (32 causal features).
- **Split protocols:** `P1-random` (5 packet-level folds), `P2-device`
  (8 device-disjoint folds), and `P3-segment` (3 segment holdouts).
- **Classifiers:** logistic regression, Random Forest, histogram gradient
  boosting, and an MLP. The output includes 192 fold/model/regime records.

The primary deployment-oriented comparison is Random Forest on `F-BEHAV`
under `P2-device`. Reported fold-mean macro-F1 is **0.7843**, compared with
**0.9984** for `P1-random` with `F-ALL`. Identifier-only probes achieve
**0.9927** macro-F1 in the reported shortcut audit. These figures are specific
to the sampled benchmark and its evaluation protocols.

## Repository layout

```text
src/gotham_ids_pipeline.py             Full experiment pipeline
scripts/verify_results.py             Offline checks of the supplied exports
results/metrics.csv                   Per-fold model metrics (192 rows)
results/per_class.csv                 Per-fold, per-class scores
results/confusion_matrices.zip        192 original .npz confusion matrices
results/folds.json                    Fold counts and train/test sizes
results/fold_composition.csv          Held-out device and segment composition
results/tables/                      Summary tables and pooled per-class scores
results/figures/                     Figures exported during the reported run
results/RUN_REPORT.md                 Original execution summary
results/environment.txt               Recorded software versions and seed
data/README.md                         Source dataset and sampling notes
requirements.txt                       Python dependencies
CITATION.cff                           Citation metadata for this code release
```

The confusion matrices are archived solely to simplify browser-based GitHub
uploads. They remain individual, unchanged `.npz` files inside
`results/confusion_matrices.zip`. Extract this archive to restore the original
`confusion/` directory.

## Verify the published result files

Install Python dependencies, then run the offline verifier, which does **not**
download the source dataset or retrain any model:

```bash
python -m pip install -r requirements.txt
python scripts/verify_results.py
```

It checks the 192 model/fold combinations and confusion matrices, test-set
sizes, the 404,324-row pooled device-disjoint result, key headline values,
and the checksum manifest. The checks do not replace a full experimental re-run.

## Re-run the experiments

The source pipeline originated as a Google Colab workflow and remains
compatible with it. Use a runtime with sufficient disk space for the complete
Gotham device-level CSVs and, optionally, a GPU for neural models.

1. Open Google Colab and upload `src/gotham_ids_pipeline.py` to the session.
2. In the file's configuration section, use `SCALE = "pilot"` for the first run.
   Set `USE_DRIVE = True` to cache the downloaded data and experiment output in
   your own Google Drive, or set it to `False` for local session storage.
3. Execute the uploaded file in a code cell:

   ```python
   %run /content/gotham_ids_pipeline.py
   ```

4. Stop if any `GATE: FAIL` is reported and investigate the input or runtime.
5. After successful pilot checks, set `SCALE = "main"` and re-run to reproduce
   the manuscript configuration. The `main` setting retains up to 8,000
   records per device with random seed `20260101`.

`SCALE = "full"` is a separate, higher-cap experiment; it is **not** the run
used for the submitted tables. The full matrix involves many hours of fitting
and requires significant memory and disk capacity. Execution time varies by
hardware and dataset-release availability.

This script imports Google Colab helpers when using Drive or initiating a
browser download; these helpers are optional at runtime. It also uses a
local filesystem path rooted at `/content` by default, so change `LOCAL_DIR`
when executing outside Colab.

## Interpretation of included tables

- `metrics.csv` contains all model-level validation scores at fold granularity.
- `per_class.csv` is **per fold**, not pooled across all eight `P2` test sets.
- `tables/table4_per_class.csv` contains *unweighted means of per-fold scores*;
  it is not the manuscript's pooled class table.
- `tables/pooled_p2_behav_rf_per_class.csv` is recalculated from the eight
  `P2-device` / `F-BEHAV` / Random Forest confusion matrices and supports
  verification of the manuscript's pooled class-level comparison.
- `figures/` and `tables/` retain their names assigned by the experiment
  pipeline; these numbers are not necessarily the final manuscript numbering.
- `fl_rounds.csv` holds exploratory federated-learning results and is distinct
  from the primary comparison reported in the manuscript.

## Data source and attribution

Belarbi, O., Spyridopoulos, T., Anthi, E., Rana, O., Carnelli, P., and Khan, A.
(2025). *A Device-Level IoT Network Traffic Dataset with Distributed Capture
and Non-IID Characteristics*. Zenodo.
https://doi.org/10.5281/zenodo.14502760

Related data article: Belarbi et al. (2026), *Data*, **11**(8), 207.
https://doi.org/10.3390/data11080207

The original public dataset is not included here. The `results/` directory
contains experimental summaries, figures and confusion matrices derived from
that dataset. Please cite the dataset authors when using these resources.

## Scope and limitations

The measured generalization is to held-out simulated devices/segments from the
Gotham testbed, not to independently collected production networks. The
source dataset and package versions may change. The supplied results were
validated for structure and internal consistency, but the full multi-hour
training workflow has not been re-executed as part of this repository release.

The repository owner should select an explicit software license before
releasing it publicly. No license grant is implied by making the repository
publicly accessible.
