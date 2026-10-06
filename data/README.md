# data/

Put the real HR exports here (this folder is git-ignored, except this file):

- `Career_Transaction_Karyawan_2012-2026_excel.xlsx`
- `Dataset.xlsx`
- optional `jobfamily_map.xlsx`, `hris_demographics.xlsx`

File names and paths are set in `config.yaml`. Column layout: `docs/data_dictionary.md`.

No real data? `python -m rotation_fit synthetic --out data/synthetic` writes fake
files with the same layout.
