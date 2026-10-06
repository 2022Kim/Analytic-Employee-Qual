# Analytic-Employee-Qual — Rotation Fit

Answers which factors really influence a rotation's success, and estimates the
**probability that an employee succeeds if rotated into a specific role**.

```
rotation_fit/      the pipeline as a package (one source of truth)
  data.py          load + validate the Excel exports
  cohort.py        who counts as rotated, and the success_fair label
  features.py      EQS scoring + similarity to the destination team
  models.py        dummy / logistic regression / random forest / ANN pipelines
  significance.py  likelihood-ratio tests: which attributes are significant
  train.py         hold-out, experiment grid, importance, final calibrated model
  predict.py       score employee x role pairs, rank candidates, explain
  report.py        shareable aggregate report  |  local model + cohort files
app/               Streamlit webapp for the internal team
notebooks/         thin notebook (thesis narrative); legacy/ = original notebook
tests/             pytest suite, runs on synthetic data
config.yaml        every setting in one place
docs/data_dictionary.md   the columns the code expects
```

## Quick start

```bash
pip install -r requirements.txt          # Python 3.10+
# put the Excel exports in data/  (paths are in config.yaml)
python -m rotation_fit train             # cohort -> experiment -> model + report
streamlit run app/streamlit_app.py       # the webapp
```

No real data at hand? `python -m rotation_fit synthetic --out data/synthetic`, or pick
**Synthetic demo data** in the app's sidebar.

Score a spreadsheet of proposed rotations (columns: `NIK`, `destination_position`,
`destination_org_unit`, optional `rotation_date`; template in `templates/`):

```bash
python -m rotation_fit score --pairs proposed.xlsx --out outputs/scored.xlsx
```

## The webapp

For the internal team; data is read from the Excel exports (paths in `config.yaml`, or
uploaded in the sidebar) and kept in memory only.

* **Score a rotation** — employee + destination position/unit → success %, an
  "unusual case" warning, and what moves the estimate up or down.
* **Rank candidates for a role** — everyone (or selected Divisi) ranked for one role;
  download as Excel.
* **Batch (spreadsheet)** — upload proposed pairs, download the scored file.
* **Model** — hold-out metrics, significant attributes, limitations, and a retrain button.

To share it with the team, run it on one internal machine
(`streamlit run app/streamlit_app.py --server.address 0.0.0.0`) behind the company
network / VPN. Don't deploy it to a public host: it shows personal data.

## What is safe to commit or share

| path | content | commit? |
|---|---|---|
| `data/`, `*.xlsx` | raw HR exports | **never** (git-ignored) |
| `outputs/` | model file + engineered cohort | **never** (git-ignored) |
| `reports/latest/` | counts, metrics, charts — no row-level data, small groups suppressed | yes, after a look |

To share results with a reviewer (or with Claude), commit `reports/latest/`.

## Method, in brief

1. **Cohort**: last valid `Movement` per employee between 2024-10-01 and 2025-03-31 where
   position or unit actually changed; promotions, acting roles and outsourcing excluded.
2. **Label** `success_fair`: the 2025 rating beat the average 2025 rating of everyone who
   started 2024 at the same level, which removes the company-wide recalibration.
3. **Features**: the EQS dimensions (kinerja, kompetensi, rekam jejak, pelatihan,
   sertifikasi, aspirasi, pengalaman), age, education, job level, job family change, and
   Gower similarity to the destination team as it was on the rotation date.
4. **Evaluation**: a 20 % stratified hold-out is set aside first. The split-ratio
   (70/75/80 %) × seed × scaler grid runs on the rest; the best model is chosen there and
   evaluated on the hold-out **once**.
5. **Significant attributes**: drop-one likelihood-ratio tests (Holm-corrected) on the
   development rows. A model using only those is compared with the full model, with the
   selection redone inside every split; it is deployed only if it costs at most
   `reduced_tolerance` of the metric.
6. **Final model**: refitted on all labelled rows and calibrated (sigmoid), with an
   out-of-distribution flag for pairings unlike anything seen in training.

### Fixed compared with the original notebook

* The saved model ignored the best split ratio and was trained on 70 % of the data.
* Choosing the model on the same test splits used to report it inflated the score;
  there is now an untouched hold-out set.
* HRIS gender/marital were merged into the cohort only, so similarity never used them.
* `edu_years` matched substrings: "S2 **Ma**najemen" scored as SMA-level "ma" (12 years).
* Isotonic calibration on ~150 positives overfits → sigmoid.
* Importance was taken from one split with one-hot columns shuffled separately; now raw
  attributes, across splits, plus a real significance test.
* The sklearn ANN had no class weights (unlike Keras / RF / logreg), and the ANN now
  tunes its size and weight decay on training rows only.
* `score_new()` depended on notebook globals and could not score a new pair; feature
  engineering is now reusable, and the TF-IDF vocabulary, similarity scale and OOD
  reference are stored with the model.
* The OOD threshold counted each reference row as its own neighbour.
* `warnings.filterwarnings("ignore")` hid real problems; removed.

## Development

```bash
pip install -r requirements-dev.txt
pytest                    # ~1-2 minutes, synthetic data only
```

Optional Keras ANN: `pip install tensorflow` and set `experiment.ann_backend: keras`.
In Google Colab: upload the repo folder (or mount Drive), `%cd` into it,
`!pip install -r requirements.txt`, then open `notebooks/rotation_fit.ipynb`.
