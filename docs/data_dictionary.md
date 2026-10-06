# Data dictionary (draft — please confirm the ❓ items)

This is a description of the **columns** the code reads — no actual data. It lets
anyone build, test and review the code with fake data (`python -m rotation_fit synthetic`)
without seeing a real employee record.

Column names must match exactly (leading/trailing spaces are ignored). Rows marked
*required* make the loader stop with a clear message when missing; *optional* columns
are treated as empty.

## 1. `Dataset.xlsx` — one row per employee

| column | type | required | used for | notes |
|---|---|---|---|---|
| `NIK` | text | ✔ | key for every join | Not purely numeric (e.g. `1486.020-R`); must be unique |
| `Nama` | text | | shown in the app only | **never** a model input; names are not unique |
| `Kinerja 2024` | rating label | ✔ | pre-rotation rating → `eqs_kinerja`; label baseline | labels in `config.yaml → ratings.order` |
| `Kinerja 2025` | rating label | ✔ | post-rotation rating → training label; latest rating when scoring a new rotation | |
| `Job Fit` | number | ✔ | `eqs_kompetensi` | recorded **before** the rotation (confirmed). ❓ 0–1 ratio or 0–100? The code accepts both |
| `Tgl Lahir` | date | ✔ | `usia` (age at rotation) | |
| `Level Pendidikan 1` | text | ✔ | `pendidikan_years` | recognised: SMA/SMU/SMK/STM/SLTA/MA/Paket C, D1–D4, S1, Profesi, S2, S3. ❓ any other spellings (e.g. "Sarjana", "D-III")? |
| `Level Jabatan` | category | | model input | recorded **before** the rotation (confirmed) |
| `Divisi` | text | ✔ | candidate filter in the app | |
| `Group` | text | ✔ | similarity fallback when the destination team has < 4 people | ❓ current (post-Sept-2025) structure? |
| `Job Fam` | text | ✔ | employee job family when no mapping file is given | ❓ current or pre-rotation? |
| `Catatan Disiplin (Aktif)` | text | | `eqs_rekam_jejak` = 0 when filled | blank = no record |
| `Catatan Disiplin (Non-aktif)` | text | | `eqs_rekam_jejak` = 50 when filled | blank = no record |
| `Pelatihan 1` … `Pelatihan 5` | text | | `eqs_pelatihan` (relevance to the target position) | |
| `Sertifikasi 1` … `Sertifikasi 3` | text | | `eqs_sertifikasi` | |
| `Aspirasi Individual` / `Jobholder` / `Unit` / `Supervisor` | text, one position per line (`- Tax Analyst`) | | `eqs_aspirasi` = 20 / 25 / 25 / 30 | highest matching column wins |

## 2. `Career_Transaction_Karyawan_2012-2026_excel.xlsx` — one row per career event

The export has **4 title rows** above the header (`config.yaml → paths.career_skiprows`).

| column | type | required | used for |
|---|---|---|---|
| `Employee No` | text | ✔ | same ID as `NIK` |
| `Start Date` | date | ✔ | rotation date; historical team rosters |
| `Career Transition` | category | ✔ | only `Movement` rows count as rotations |
| `Transition Type` | category | ✔ | excludes Promotion, Demotion, Pelaksana Tugas/Harian, Pejabat/Pengganti Sementara, contract and status changes. ❓ full list of values? |
| `Employment Status`, `Old Employment Status` | category | ✔ | excludes `Outsource` |
| `Old Position`, `Position` | text | ✔ | origin / destination position |
| `Old Organization Unit`, `Organization Unit` | text | ✔ | origin / destination team |
| `Old Grade`, `Grade` | text | | carried along, not a model input |

## 3. Optional: job family mapping (`paths.jobfamily_map`)

Two columns, any header: **position**, **job_family**. Enables `job_family_changed`,
`eqs_pengalaman` (years in the destination job family) and job family in similarity.

## 4. Optional: HRIS demographics (`paths.hris_demographics`)

Columns named in `config.yaml → hris_columns` (default `NIK`, `gender`, `marital`).
Used for similarity and the subgroup fairness check.

## 5. Spreadsheet for batch scoring (`templates/score_pairs_template.csv`)

| column | required | notes |
|---|---|---|
| `NIK` | ✔ | employee to evaluate |
| `destination_position` | ✔ | target role |
| `destination_org_unit` | ✔ | target team, spelled as in the career file |
| `rotation_date` | | default: today |
| `destination_group` | | only needed when the destination team is new or tiny |
