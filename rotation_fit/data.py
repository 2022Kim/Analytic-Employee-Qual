"""Load and validate the HR spreadsheets.

Every loader accepts a path or a file-like object (e.g. a Streamlit upload),
in .xlsx / .xls / .csv format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REQUIRED_CAR = ["Employee No", "Start Date", "Career Transition", "Transition Type",
                "Employment Status", "Old Employment Status",
                "Old Position", "Position",
                "Old Organization Unit", "Organization Unit"]
OPTIONAL_CAR = ["Old Grade", "Grade"]

REQUIRED_EMP = ["NIK", "Job Fit", "Tgl Lahir", "Level Pendidikan 1",
                "Divisi", "Group", "Job Fam"]


def norm(s: pd.Series) -> pd.Series:
    """Case- and whitespace-insensitive text for comparisons."""
    return s.astype("string").str.strip().str.casefold()


def key(s: pd.Series) -> pd.Series:
    """IDs are not purely numeric — e.g. '1486.020-R' — so keep them as text."""
    out = s.astype("string").str.strip()
    # Excel turns numeric IDs into floats ("1234.0"); undo that.
    return out.str.replace(r"^(\d+)\.0$", r"\1", regex=True)


def norm_value(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip().casefold()


def read_table(src, skiprows: int = 0) -> pd.DataFrame:
    name = getattr(src, "name", src)
    suffix = Path(str(name)).suffix.lower()
    if hasattr(src, "seek"):
        src.seek(0)
    if suffix == ".csv":
        df = pd.read_csv(src, skiprows=skiprows)
    else:
        df = pd.read_excel(src, skiprows=skiprows)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _require(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}")


def load_career(src, skiprows: int = 4) -> pd.DataFrame:
    car = read_table(src, skiprows=skiprows)
    _require(car, REQUIRED_CAR, "career file")
    for c in OPTIONAL_CAR:
        if c not in car.columns:
            car[c] = pd.NA
    car["Start Date"] = pd.to_datetime(car["Start Date"], errors="coerce")
    car["_k"] = key(car["Employee No"])
    return car


def load_dataset(src, cfg: dict) -> pd.DataFrame:
    emp = read_table(src)
    r = cfg["ratings"]
    _require(emp, REQUIRED_EMP + sorted({r["before"], r["after"], r["current"]}), "dataset")
    emp["_k"] = key(emp["NIK"])
    # Employee Name is NOT unique — always key on the ID.
    dup = emp["_k"][emp["_k"].duplicated()].unique()
    if len(dup):
        raise ValueError(f"NIK is not unique in the dataset ({len(dup)} duplicated IDs)")
    for col in (cfg["eqs"]["pelatihan_columns"] + cfg["eqs"]["sertifikasi_columns"]
                + list(cfg["eqs"]["aspirasi_points"])
                + ["Catatan Disiplin (Aktif)", "Catatan Disiplin (Non-aktif)", "Level Jabatan"]):
        if col not in emp.columns:
            emp[col] = pd.NA
    if "Nama" not in emp.columns:
        emp["Nama"] = pd.NA
    return emp


def load_jobfamily_map(src) -> dict:
    jf = read_table(src)
    if jf.shape[1] < 2:
        raise ValueError("job family mapping needs 2 columns: position, job_family")
    return dict(zip(norm(jf.iloc[:, 0]), jf.iloc[:, 1]))


def load_demographics(src, cfg: dict) -> pd.DataFrame:
    cols = cfg["hris_columns"]
    dg = read_table(src)
    _require(dg, [cols["id"]], "HRIS demographics")
    out = pd.DataFrame({"_k": key(dg[cols["id"]])})
    for std in ("gender", "marital"):
        if cols.get(std) in dg.columns:
            out[std] = dg[cols[std]].astype("string").str.strip()
    return out.drop_duplicates("_k")


@dataclass
class HRData:
    career: pd.DataFrame
    emp: pd.DataFrame
    jf_map: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def emp_by_key(self) -> pd.DataFrame:
        return self.emp.set_index("_k", drop=False)


def load_hr_data(cfg: dict, career=None, dataset=None, jobfamily_map=None,
                 demographics=None) -> HRData:
    """Load everything. Arguments override the paths in the config."""
    p = cfg["paths"]
    car = load_career(career if career is not None else p["career"], p["career_skiprows"])
    emp = load_dataset(dataset if dataset is not None else p["dataset"], cfg)
    notes = []

    jf_src = jobfamily_map if jobfamily_map is not None else p.get("jobfamily_map")
    jf_map = load_jobfamily_map(jf_src) if jf_src else None
    if jf_map is None:
        notes.append("No job family mapping: origin/destination job family, "
                     "job_family_changed and eqs_pengalaman are unavailable.")

    dg_src = demographics if demographics is not None else p.get("hris_demographics")
    if dg_src:
        # Merge into emp (not only the cohort) so similarity can compare the
        # employee with the destination team on these dimensions too.
        emp = emp.drop(columns=[c for c in ("gender", "marital") if c in emp.columns])
        emp = emp.merge(load_demographics(dg_src, cfg), on="_k", how="left")
    else:
        notes.append("No HRIS demographics: gender and marital status unavailable.")

    return HRData(career=car, emp=emp, jf_map=jf_map, notes=notes)
