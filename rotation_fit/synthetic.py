"""Fake HR data with exactly the layout of the real exports.

Used by the tests and for trying the app without real employee data. Every
name, ID and record is randomly generated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RATINGS = ["Need Improvement", "Satisfactory", "Successful", "Excellent", "Outstanding"]
EDU = ["SMA", "D3", "S1", "S1", "S1", "S2", "S3"]
LEVELS = ["Staff", "Officer", "Supervisor", "Manager"]
ROLES = ["Analyst", "Officer", "Specialist", "Coordinator"]
TOPICS = {
    "Finance": ["accounting", "tax", "budget", "treasury"],
    "Human Capital": ["recruitment", "payroll", "training", "talent"],
    "Operations": ["logistics", "warehouse", "maintenance", "safety"],
    "IT": ["network", "software", "data", "security"],
    "Marketing": ["brand", "digital", "sales", "market research"],
}


def make_synthetic(out_dir, n_employees: int = 900, seed: int = 0,
                   with_demographics: bool = True, with_jobfamily: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- organisation -------------------------------------------------
    units, positions, jf_rows = [], [], []
    for fam, words in TOPICS.items():
        for i, w in enumerate(words):
            unit = f"{fam} - {w.title()} Unit"
            units.append({"unit": unit, "group": f"Group {fam}", "divisi": f"Divisi {fam}",
                          "family": fam, "topic": w})
            for r in ROLES:
                pos = f"{w.title()} {r}"
                positions.append({"position": pos, "unit": unit, "family": fam, "topic": w})
                jf_rows.append({"position": pos, "job_family": fam})
    units = pd.DataFrame(units)
    positions = pd.DataFrame(positions)

    # ---- employees ----------------------------------------------------
    n = n_employees
    nik = [f"{1000 + i}.{rng.integers(0, 999):03d}" + ("-R" if rng.random() < 0.05 else "")
           for i in range(n)]
    home = positions.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    birth = pd.Timestamp("1965-01-01") + pd.to_timedelta(rng.integers(0, 365 * 35, n), "D")
    k24 = rng.choice(5, n, p=[0.05, 0.15, 0.45, 0.25, 0.10])
    job_fit = np.clip(rng.normal(0.72, 0.12, n), 0.2, 1.0).round(2)
    disc_active = rng.random(n) < 0.03
    disc_past = ~disc_active & (rng.random(n) < 0.08)

    # ---- career history ----------------------------------------------
    car_rows = []
    final_pos = home.copy()
    rotated_to = [None] * n
    for i in range(n):
        hire = pd.Timestamp("2012-01-01") + pd.to_timedelta(int(rng.integers(0, 365 * 11)), "D")
        car_rows.append(dict(k=nik[i], date=hire, ct="Hire", tt="New Hire", st="Permanent",
                             ost="", op="", p=home.position[i], ou="", u=home.unit[i]))
        r = rng.random()
        if r < 0.55:   # rotation inside the study window
            dest = positions.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            if rng.random() < 0.5:   # half the time stay in the same family
                same = positions[positions.family == home.family[i]]
                dest = same.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            d = pd.Timestamp("2024-10-01") + pd.to_timedelta(int(rng.integers(0, 180)), "D")
            ttype = rng.choice(["Rotasi", "Mutasi", "Promotion", "Pelaksana Tugas"],
                               p=[0.6, 0.25, 0.1, 0.05])
            status = "Outsource" if rng.random() < 0.03 else "Permanent"
            car_rows.append(dict(k=nik[i], date=d, ct="Movement", tt=ttype, st=status,
                                 ost=status, op=home.position[i], p=dest.position,
                                 ou=home.unit[i], u=dest.unit))
            final_pos.loc[i] = dest
            rotated_to[i] = dest
        elif r < 0.65:  # older rotation, outside the window
            dest = positions.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            d = pd.Timestamp("2019-01-01") + pd.to_timedelta(int(rng.integers(0, 1500)), "D")
            car_rows.append(dict(k=nik[i], date=d, ct="Movement", tt="Rotasi", st="Permanent",
                                 ost="Permanent", op=home.position[i], p=dest.position,
                                 ou=home.unit[i], u=dest.unit))
            final_pos.loc[i] = dest

    # ---- aspirations, training and the hidden outcome -----------------
    asp = {c: [None] * n for c in ["Aspirasi Individual", "Aspirasi Jobholder",
                                   "Aspirasi Unit", "Aspirasi Supervisor"]}
    pel = {f"Pelatihan {j}": [None] * n for j in range(1, 6)}
    ser = {f"Sertifikasi {j}": [None] * n for j in range(1, 4)}
    signal = np.zeros(n)
    for i in range(n):
        dest = rotated_to[i]
        target = dest if dest is not None else home.iloc[i]
        if rng.random() < 0.35:
            col = rng.choice(list(asp))
            asp[col][i] = f"- {target.position}\n- {rng.choice(positions.position)}"
            signal[i] += 0.5 if dest is not None else 0
        topic = target.topic if rng.random() < 0.5 else rng.choice(sum(TOPICS.values(), []))
        relevant = topic == target.topic
        for j in range(1, int(rng.integers(1, 6))):
            pel[f"Pelatihan {j}"][i] = f"Workshop {topic} {rng.choice(['basic', 'advanced'])}"
        if rng.random() < 0.4:
            ser["Sertifikasi 1"][i] = f"Certified {topic} Professional"
        signal[i] += 0.4 * relevant
        if dest is not None and dest.family != home.family[i]:
            signal[i] -= 0.4
    signal += 3.0 * (job_fit - 0.72) - 0.8 * disc_active - 0.3 * disc_past
    # 2025 is stricter for everyone (recalibration) plus the rotation effect
    k25 = np.clip(np.round(k24 - 0.4 + signal + rng.normal(0, 0.7, n)), 0, 4).astype(int)

    emp = pd.DataFrame({
        "NIK": nik, "Nama": [f"Employee {i:04d}" for i in range(n)],
        "Kinerja 2024": [RATINGS[v] for v in k24], "Kinerja 2025": [RATINGS[v] for v in k25],
        "Job Fit": job_fit, "Tgl Lahir": birth,
        "Level Pendidikan 1": rng.choice(EDU, n),
        "Level Jabatan": rng.choice(LEVELS, n, p=[0.5, 0.25, 0.15, 0.10]),
        "Divisi": final_pos.unit.map(units.set_index("unit").divisi),
        "Group": final_pos.unit.map(units.set_index("unit").group),
        "Job Fam": final_pos.family, "Position": final_pos.position,
        "Catatan Disiplin (Aktif)": np.where(disc_active, "SP1", None),
        "Catatan Disiplin (Non-aktif)": np.where(disc_past, "SP1 2021", None),
        **pel, **ser, **asp,
    })
    car = pd.DataFrame(car_rows).rename(columns={
        "k": "Employee No", "date": "Start Date", "ct": "Career Transition",
        "tt": "Transition Type", "st": "Employment Status", "ost": "Old Employment Status",
        "op": "Old Position", "p": "Position", "ou": "Old Organization Unit",
        "u": "Organization Unit"})
    car["Old Grade"], car["Grade"] = "G5", "G5"

    paths = {"dataset": out / "Dataset.xlsx", "career": out / "Career_Transaction.xlsx"}
    emp.to_excel(paths["dataset"], index=False)
    # the real export has 4 title rows above the header
    with pd.ExcelWriter(paths["career"]) as w:
        pd.DataFrame([["Career Transaction Report (SYNTHETIC)"], [""], [""], [""]]) \
            .to_excel(w, index=False, header=False)
        car.to_excel(w, index=False, startrow=4)
    if with_jobfamily:
        paths["jobfamily_map"] = out / "jobfamily_map.xlsx"
        pd.DataFrame(jf_rows).to_excel(paths["jobfamily_map"], index=False)
    if with_demographics:
        paths["hris_demographics"] = out / "hris_demographics.xlsx"
        pd.DataFrame({"NIK": nik, "gender": rng.choice(["M", "F"], n),
                      "marital": rng.choice(["Married", "Single"], n, p=[0.6, 0.4])}) \
            .to_excel(paths["hris_demographics"], index=False)
    return paths


def synthetic_config_overrides(paths: dict) -> dict:
    return {"paths": {"career": str(paths["career"]), "career_skiprows": 4,
                      "dataset": str(paths["dataset"]),
                      "jobfamily_map": str(paths.get("jobfamily_map") or "") or None,
                      "hris_demographics": str(paths.get("hris_demographics") or "") or None}}
