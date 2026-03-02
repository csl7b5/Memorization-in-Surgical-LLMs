import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import re
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
from config import CSV_PATH as INPUT_CSV, CARDS_DIR
ENCODING = "cp1252"  # your file reads with cp1252

OUT_DIR = Path(CARDS_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: if you have an internal patient identifier column you trust, set it here.
# If None, we will use the row index as patient_id.
PATIENT_ID_COL = None  # e.g., "patient_id" if you have one

# -----------------------------
# MAPPINGS (from your schema)
# -----------------------------
ANEURYSM_INVOLVEMENT_MAP = {
    0: "None",
    1: "Root",
    2: "Ascending",
    3: "Arch",
    4: "Descending",
    5: "Abdominal",
}

AAS_MAP = {
    0: "None",
    1: "Acute aortic syndrome: Type A (ascending) dissection",
    2: "Acute aortic syndrome: Type B (descending) dissection",
    3: "Acute aortic syndrome: Intramural hematoma",
    4: "Acute aortic syndrome: Penetrating atherosclerotic ulcer (PAU)",
}

COMPLICATING_FACTORS_MAP = {
    0: "None",
    1: "Rupture",
    2: "Cardiac tamponade",
    3: "Malperfusion (e.g., MI, stroke, paraplegia, abdominal pain, AKI, absent/unequal pulses)",
    4: "Other complicating factor",
}

CAUSE_OF_DEATH_MAP = {
    1: "Aortic/Cardiac",
    2: "Other",
}

# Pathology code map (your list)
PATHOLOGY_MAP = {
    1: "Medial degeneration",
    2: "Cystic medial degeneration",
    3: "Myxoid material accumulation/degeneration",
    4: "Atherosclerosis",
    5: "Hemorrhage",
    6: "Mucoid material accumulation",
    7: "Hyalinization",
    8: "Calcification",
    9: "Elastin fragmentation",
    10: "Fibrosis",
    11: "Inflammation",
    12: "Medionecrosis",
    13: "Dissection",
    14: "Granuloma",
    15: "Fibrin",
    16: "Thrombosis",
    17: "Pseudoneointima",
    18: "Lamellar collapse",
    19: "No abnormality",
    20: "Intimal hyperplasia",
    21: "Neovascularization of tunica media",
    22: "Smooth muscle disarray",
    23: "Pseudoaneurysm",
    24: "Intimal thickening",
}

# Surgery columns you have
SURG_TYPES = [
    "aortic_valve_repair",
    "aortic_valve_replacement",
    "aortic_root_repair",
    "aortic_root_replacement",
    "ascending_aorta_replacement",
    "hemiarch_replacement",
    "total_arch_replacement",
    "stage_I_elephant_trunk",
    "stage_II_elephant_trunk",
    "TEVAR",
    "CABG",
    "descending_replacement",
]

# Group surgery flags into clinician-friendly categories
SURG_CATEGORY_RULES = [
    ("Aortic root repair", ["aortic_root_repair"]),
    ("Aortic root replacement", ["aortic_root_replacement"]),
    ("Ascending aorta replacement", ["ascending_aorta_replacement"]),
    ("Hemiarch replacement", ["hemiarch_replacement"]),
    ("Total arch replacement", ["total_arch_replacement"]),
    ("Elephant trunk (stage I)", ["stage_I_elephant_trunk"]),
    ("Elephant trunk (stage II)", ["stage_II_elephant_trunk"]),
    ("TEVAR", ["TEVAR"]),
    ("Descending aorta replacement", ["descending_replacement"]),
    ("CABG", ["CABG"]),
    ("Aortic valve repair", ["aortic_valve_repair"]),
    ("Aortic valve replacement", ["aortic_valve_replacement"]),
]

# -----------------------------
# HELPERS
# -----------------------------
def _to_int_list(x) -> list[int]:
    """
    Parse strings like "1, 2, 3" or "1,2" or numeric into list[int].
    If x contains other text like "4 (atrial fibrillation)", we extract leading digits.
    """
    if pd.isna(x):
        return []
    if isinstance(x, (int, float)) and not np.isnan(x):
        return [int(x)]
    s = str(x).strip()
    if not s or s == "\xa0":  # non-breaking space
        return []
    digits = re.findall(r"\d+", s)
    return [int(d) for d in digits] if digits else []

def _safe_bool01(x) -> int:
    if pd.isna(x):
        return 0
    try:
        return 1 if float(x) == 1.0 else 0
    except Exception:
        s = str(x).strip().lower()
        return 1 if s in ("1", "yes", "y", "true") else 0

def _age_bucket(age):
    if age is None or pd.isna(age):
        return "Unknown"
    if isinstance(age, str):
        age = age.replace("\xa0", "").strip()
        if not age:
            return "Unknown"
    try:
        a = int(float(age))
    except Exception:
        return "Unknown"
    lo = (a // 10) * 10
    hi = lo + 9
    return f"{lo}–{hi}"

def _diameter_bucket(mm):
    if mm is None or pd.isna(mm):
        return "Unknown"
    if isinstance(mm, str):
        mm = mm.replace("\xa0", "").strip()
        if not mm:
            return "Unknown"
    try:
        v = float(mm)
    except Exception:
        return "Unknown"
    if v < 50:
        return "<50 mm"
    if v < 60:
        return "50–59 mm"
    return "≥60 mm"

def _clean_free_text(s: str) -> str:
    """
    Keep clinically useful indication text but scrub common date-like patterns.
    """
    if s is None or pd.isna(s):
        return ""
    t = str(s).strip()
    if not t or t == "\xa0":
        return ""
    if t.lower() in ("0", "0.0", "none", "n/a", "na", "nan"):
        return ""
    # remove obvious dates like 2020-01-02, 01/02/2020, etc.
    t = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[DATE]", t)
    t = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]", t)
    # remove long MRN-like digit strings
    t = re.sub(r"\b\d{7,}\b", "[ID]", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t

def _map_multi(code_val, mapping):
    codes = _to_int_list(code_val)
    if not codes:
        return []
    out = []
    for c in codes:
        if c in mapping:
            out.append(mapping[c])
        else:
            out.append(f"Unknown({c})")
    # de-dupe while preserving order
    seen = set()
    deduped = []
    for item in out:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped

def summarize_surgery(row, n: int, mode: str = "full"):
    """
    Summarize surgery n as:
      - age at surgery (read directly from surg_n_age)
      - categories based on surg_n_* flags
      - free text 'surg_n_type' and 'surg_n_others' (scrubbed)
    """
    age_col = f"surg_{n}_age"
    type_col = f"surg_{n}_type"
    other_col = f"surg_{n}_others"

    # Try reading the age integer directly
    age_val = row.get(age_col)
    age_at = None
    if not pd.isna(age_val):
        try:
            if mode == "exact":
                age_at = round(float(age_val), 2)
            else:
                age_at = int(float(age_val))
        except Exception:
            pass

    # Flag-driven categories
    cats = []
    for label, keys in SURG_CATEGORY_RULES:
        for k in keys:
            col = f"surg_{n}_{k}"
            if col in row.index and _safe_bool01(row.get(col)) == 1:
                cats.append(label)
                break

    cats = list(dict.fromkeys(cats))  # preserve order, dedupe

    # Additional surgery type / others fields
    s_type = _clean_free_text(row.get(type_col))
    s_other = _clean_free_text(row.get(other_col))

    # Build a clinician-readable line
    age_str = f"age {age_at}" if age_at is not None else "age unknown"
    core = ", ".join(cats) if cats else "Aortic surgery (type unspecified)"

    extras = []
    if s_type:
        extras.append(f"type: {s_type}")
    if s_other:
        extras.append(f"other: {s_other}")

    extra_str = f" ({'; '.join(extras)})" if extras else ""
    return age_at, f"{ordinal(n)} surgery ({age_str}): {core}{extra_str}."

def ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")

def build_card(row, mode: str = "full") -> str:
    """
    mode in {"full", "partial", "coarsened"}
    """
    sex = str(row.get("Sex")).strip() if not pd.isna(row.get("Sex")) else "Unknown"
    age = row.get("age")
    age_int = None
    try:
        if not pd.isna(age):
            if isinstance(age, str):
                age = age.replace("\xa0", "").strip()
            if age:
                if mode == "exact":
                    age_int = round(float(age), 2)
                else:
                    age_int = int(float(age))
    except Exception:
        age_int = None

    fam_hx = _safe_bool01(row.get("fam_hx"))

    # Genetics
    pathogenic_gene = "" if pd.isna(row.get("Pathogenic Gene")) else str(row.get("Pathogenic Gene")).strip()
    vus_gene = "" if pd.isna(row.get("VUS Gene")) else str(row.get("VUS Gene")).strip()

    # Clinical presentation
    aneurysm_sites = _map_multi(row.get("Aneurysm_involvement"), ANEURYSM_INVOLVEMENT_MAP)
    aas = _map_multi(row.get("Acute_aortic_syndrome"), AAS_MAP)
    er = _safe_bool01(row.get("ER_presentation"))
    complicating = _map_multi(row.get("Complicating_factor"), COMPLICATING_FACTORS_MAP)

    # Valve anatomy
    bav = _safe_bool01(row.get("Bicuspid_aortic_valve"))

    # Diameters
    first_diam = row.get("first_reported_diameter")
    interv_diam = row.get("intervention_diameter")

    # Pathology
    path_codes = []
    for col in ["pathology_v1", "pathology_v2"]:
        if col in row.index:
            path_codes.extend(_to_int_list(row.get(col)))
    path_labels = []
    for c in path_codes:
        if c in PATHOLOGY_MAP:
            path_labels.append(PATHOLOGY_MAP[c])
    path_labels = list(dict.fromkeys(path_labels))

    # Surgeries
    surgery_lines = []
    surgery_ages = []
    n_surg = 0
    for n in [1, 2, 3]:
        has_age = not pd.isna(row.get(f"surg_{n}_age"))
        # count a surgery if there is an age OR any procedure flag OR a type string
        any_flag = any((_safe_bool01(row.get(f"surg_{n}_{t}")) == 1) for t in SURG_TYPES if f"surg_{n}_{t}" in row.index)
        any_type = not pd.isna(row.get(f"surg_{n}_type")) and str(row.get(f"surg_{n}_type")).strip() not in ("", "\xa0", "nan", "NaN")
        if has_age or any_flag or any_type:
            n_surg += 1
            age_at, line = summarize_surgery(row, n, mode=mode)
            surgery_lines.append(line)
            surgery_ages.append(age_at)

    underwent_reop = _safe_bool01(row.get("underwent_reoperation"))
    reop_ind = _clean_free_text(row.get("reoperation indication"))

    # Outcome
    mortality = _safe_bool01(row.get("mortality"))
    cod = _to_int_list(row.get("Causes_of_death"))
    cod_label = CAUSE_OF_DEATH_MAP.get(cod[0], "Unknown") if cod else ""

    # Build sections
    lines = []
    lines.append("Patient Summary")

    # Demographics
    if mode == "coarsened":
        age_display = _age_bucket(age_int)
    else:
        age_display = str(age_int) if age_int is not None else "Unknown"

    lines.append("")
    lines.append("Demographics:")
    lines.append(f"- Sex: {sex}")
    lines.append(f"- Age at presentation: {age_display}")
    lines.append(f"- Family history of aortic disease: {'Yes' if fam_hx else 'No/Unknown'}")

    # Genetics
    lines.append("")
    lines.append("Genetics:")
    if mode == "coarsened":
        # coarsen gene names to reduce uniqueness
        p = "Present" if pathogenic_gene else "None identified"
        v = "Present" if vus_gene else "None identified"
        lines.append(f"- Pathogenic variant: {p}")
        lines.append(f"- VUS: {v}")
    else:
        lines.append(f"- Pathogenic variant: {pathogenic_gene if pathogenic_gene else 'None identified'}")
        lines.append(f"- VUS: {vus_gene if vus_gene else 'None identified'}")

    # Clinical presentation / anatomy
    lines.append("")
    lines.append("Clinical presentation:")
    if aneurysm_sites:
        lines.append(f"- Aneurysm involvement: {', '.join(aneurysm_sites)}")
    else:
        lines.append(f"- Aneurysm involvement: Unknown/Not recorded")

    if aas and any("None" not in a for a in aas):
        # If multiple AAS codes, list them (rare but exists)
        lines.append(f"- Acute aortic syndrome: {', '.join([a.replace('Acute aortic syndrome: ', '') for a in aas if a != 'None'])}")
    else:
        lines.append("- Acute aortic syndrome: None recorded")

    if mode != "partial":
        lines.append(f"- Initial ER presentation: {'Yes' if er else 'No/Unknown'}")
        if complicating and any(c != "None" for c in complicating):
            # drop any "None" values
            cc = [c for c in complicating if c != "None"]
            # coarsen if needed
            if mode == "coarsened":
                # only indicate presence of any major complicating factor
                lines.append("- Complicating factors: Present")
            else:
                lines.append(f"- Complicating factors: {', '.join(cc)}")
        else:
            lines.append("- Complicating factors: None recorded")

    # Surgical history
    lines.append("")
    lines.append("Surgical course:")
    lines.append(f"- Number of aortic surgeries recorded: {n_surg}")

    if surgery_lines:
        if mode == "coarsened":
            # Replace exact ages in surgery lines with buckets
            bucketed = []
            for ln, age_at in zip(surgery_lines, surgery_ages):
                b = _age_bucket(age_at)
                ln2 = re.sub(r"\(age [^)]+\)", f"(age {b})", ln)
                # coarsen some specifics
                ln2 = ln2.replace("Aortic valve repair", "Valve intervention").replace("Aortic valve replacement", "Valve intervention")
                bucketed.append(ln2)
            lines.extend([f"- {x}" for x in bucketed])
        else:
            lines.extend([f"- {x}" for x in surgery_lines])
    else:
        lines.append("- No aortic surgery details recorded.")

    # Reoperation
    if mode != "partial":
        lines.append("")
        lines.append("Reoperation:")
        if underwent_reop or n_surg >= 2:
            lines.append("- Underwent reoperation: Yes")
            if reop_ind:
                if mode == "coarsened":
                    lines.append("- Indication: Progressive or residual aortic disease / other (coarsened)")
                else:
                    lines.append(f"- Indication: {reop_ind}")
            else:
                lines.append("- Indication: Not recorded")
        else:
            lines.append("- Underwent reoperation: No/Unknown")

    # Aortic size
    if mode != "partial":
        lines.append("")
        lines.append("Aortic size:")
        if mode == "coarsened":
            lines.append(f"- First reported diameter: {_diameter_bucket(first_diam)}")
            lines.append(f"- Diameter at intervention: {_diameter_bucket(interv_diam)}")
        else:
            def _format_diam(d):
                if d is None or pd.isna(d): return "Unknown"
                if isinstance(d, str):
                    d = d.replace("\xa0", "").strip()
                    if not d: return "Unknown"
                try:
                    return f"{float(d):.0f} mm"
                except Exception:
                    return "Unknown"

            lines.append(f"- First reported diameter: {_format_diam(first_diam)}")
            lines.append(f"- Diameter at intervention: {_format_diam(interv_diam)}")

    # Pathology
    if mode != "partial":
        lines.append("")
        lines.append("Histopathology:")
        if path_labels:
            if mode == "coarsened":
                lines.append("- Pathology reported: Yes (coarsened)")
            else:
                lines.append(f"- Findings: {', '.join(path_labels)}")
        else:
            lines.append("- Not recorded")

    # Valve anatomy
    if mode != "partial":
        lines.append("")
        lines.append("Valve anatomy:")
        if mode == "coarsened":
            lines.append(f"- Bicuspid aortic valve: {'Present' if bav else 'Not recorded/absent'}")
        else:
            lines.append(f"- Bicuspid aortic valve: {'Yes' if bav else 'No/Unknown'}")

    # Billing and Diagnoses (ICD-10)
    icd_raw = row.get("Icd10 Codes")
    icd_codes = []
    if not pd.isna(icd_raw):
        # Extract individual codes
        raw_list = str(icd_raw).split(",")
        for c in raw_list:
            c = c.strip()
            if c:
                icd_codes.append(c)

    if mode != "partial":
        lines.append("")
        lines.append("Billing/Diagnoses:")
        if icd_codes:
            if mode == "coarsened":
                # Coarsening strategy: truncate to base 3-character category (e.g., I71.01 -> I71)
                coarsened_codes = []
                for code in icd_codes:
                    if "." in code:
                        coarsened_codes.append(code.split(".")[0])
                    else:
                        coarsened_codes.append(code[:3])
                # Deduplicate after coarsening to avoid e.g. "I71, I71"
                coarsened_codes = list(dict.fromkeys(coarsened_codes))
                lines.append(f"- ICD-10 Codes: {', '.join(coarsened_codes)}")
            else:
                lines.append(f"- ICD-10 Codes: {', '.join(icd_codes)}")
        else:
            lines.append("- ICD-10 Codes: None recorded")

    # Outcome
    lines.append("")
    lines.append("Outcome:")
    if mortality:
        if mode == "coarsened":
            lines.append("- Vital status: Deceased")
            lines.append(f"- Cause of death category: {cod_label if cod_label else 'Unknown'}")
        else:
            lines.append("- Vital status: Deceased")
            lines.append(f"- Cause of death category: {cod_label if cod_label else 'Unknown'}")
    else:
        lines.append("- Vital status: Alive at last follow-up / not recorded as deceased")

    return "\n".join(lines).strip()


def main():
    df = pd.read_csv(INPUT_CSV, encoding=ENCODING)

    # Build patient_id
    if PATIENT_ID_COL and PATIENT_ID_COL in df.columns:
        patient_ids = df[PATIENT_ID_COL].astype(str).tolist()
    else:
        patient_ids = [f"row_{i}" for i in range(len(df))]

    outputs = {
        "full": OUT_DIR / "cards_full.jsonl",
        "partial": OUT_DIR / "cards_partial.jsonl",
        "coarsened": OUT_DIR / "cards_coarsened.jsonl",
        "exact": OUT_DIR / "cards_exact.jsonl",
    }
    # overwrite existing
    for p in outputs.values():
        if p.exists():
            p.unlink()

    # Write JSONL for each mode
    for i, row in df.iterrows():
        pid = patient_ids[i]

        # generate card types
        for mode in ["full", "partial", "coarsened", "exact"]:
            card = build_card(row, mode=mode)

            # Minimal metadata to help later experiments (no identifiers)
            meta = {
                "patient_id": pid,
                "mode": mode,
                "sex": None if pd.isna(row.get("Sex")) else str(row.get("Sex")).strip(),
                "pathogenic_gene": None if pd.isna(row.get("Pathogenic Gene")) else str(row.get("Pathogenic Gene")).strip(),
                "vus_gene": None if pd.isna(row.get("VUS Gene")) else str(row.get("VUS Gene")).strip(),
                "underwent_reoperation": int(_safe_bool01(row.get("underwent_reoperation"))),
                "mortality": int(_safe_bool01(row.get("mortality"))),
                "icd10_codes": str(row.get("Icd10 Codes")).strip() if not pd.isna(row.get("Icd10 Codes")) else None,
            }

            rec = {"meta": meta, "text": card}
            with outputs[mode].open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Also write a quick preview file
    preview = OUT_DIR / "preview_first_5_full_cards.txt"
    with preview.open("w", encoding="utf-8") as f:
        for k in range(min(5, len(df))):
            rec = build_card(df.iloc[k], mode="full")
            f.write(f"==== {k} ====\n{rec}\n\n")

    print("Wrote:")
    for mode, p in outputs.items():
        print(f"- {mode}: {p}")
    print(f"- preview: {preview}")

if __name__ == "__main__":
    main()