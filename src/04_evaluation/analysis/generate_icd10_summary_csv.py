"""
generate_icd10_summary_csv.py

Compiles all ICD-10 memorization analysis findings into CSV files:
  1. icd10_summary_overall.csv   - high-level partial-match stats per model
  2. icd10_summary_by_n_codes.csv - 100%/50%/any recall broken down by GT code count (1, 2, 3, 4, 5, 6+)
  3. icd10_summary_by_code.csv   - per clinical ICD-10 code recall (M0 vs M1 vs M2)
"""
import json, re, csv, os
from collections import defaultdict

# ── ICD-10 chapter lookup ────────────────────────────────────────────
ICD10_CHAPTERS = {
    "A": "Infectious diseases", "B": "Infectious diseases",
    "C": "Neoplasms", "D": "Neoplasms / Blood diseases",
    "E": "Endocrine / Metabolic (Diabetes, Thyroid, Obesity)",
    "F": "Mental & behavioural", "G": "Nervous system",
    "H": "Eye / Ear",
    "I": "Circulatory (HTN, Heart, Aortic)", "J": "Respiratory",
    "K": "Digestive", "L": "Skin", "M": "Musculoskeletal",
    "N": "Urinary / Renal", "Q": "Congenital malformations",
    "R": "Symptoms / Lab findings", "Z": "Supplementary / Status codes",
}

CLINICAL_HIGHLIGHTS = {
    "I10": "Essential hypertension (HTN)",
    "I11": "HTN heart disease",
    "I25": "Chronic ischemic heart disease (CAD)",
    "I35": "Aortic valve disorder",
    "I42": "Cardiomyopathy",
    "I44": "AV block",
    "I45": "Other conduction disorders",
    "I48": "Atrial fibrillation",
    "I50": "Heart failure",
    "I71": "Aortic aneurysm / dissection",
    "I73": "Peripheral vascular disease",
    "I77": "Other arterial disease / vasculopathy",
    "I80": "Phlebitis / DVT",
    "I82": "Venous thrombosis",
    "E10": "Type 1 diabetes",
    "E11": "Type 2 diabetes",
    "E78": "Dyslipidemia / Hypercholesterolemia",
    "E66": "Obesity",
    "E03": "Hypothyroidism",
    "J44": "COPD",
    "J45": "Asthma",
    "N18": "Chronic kidney disease",
    "Q21": "Congenital heart septal defect",
    "Q23": "Congenital aortic/mitral valve anomaly",
    "Q25": "Congenital aorta anomaly (bicuspid)",
    "F41": "Anxiety disorder",
    "F32": "Depression",
    "F17": "Nicotine dependence (smoking)",
    "Z82": "Family history of circulatory disease",
    "Z86": "Personal history of disease",
    "Z95": "Presence of cardiac implants",
    "Z98": "Post-procedural status",
}

def parse_gt(raw):
    if not raw or str(raw).strip().lower() in ("none recorded", "none", ""):
        return set()
    return {c.strip().upper() for c in str(raw).split(",") if c.strip()}

def extract_codes(text):
    return {c.upper() for c in re.findall(r"\b([A-Za-z]\d{2}(?:\.\d{1,4})?)\b", text)}

def norm3(c):
    return c.split(".")[0].upper()

def best_recall(gt_codes, generations):
    """Returns (recall, matched_set) for the generation with highest recall."""
    best_r, best_matched = 0.0, set()
    gt3 = {norm3(c) for c in gt_codes}
    for gen in generations:
        gen3 = {norm3(c) for c in extract_codes(gen)}
        matched = {c for c in gt_codes if norm3(c) in gen3}
        r = len(matched) / len(gt_codes) if gt_codes else 0.0
        if r > best_r:
            best_r, best_matched = r, matched
    return best_r, best_matched

def load_file(path):
    recs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            gt = parse_gt(d.get("target_icd10") or d.get("target_icd10_raw", ""))
            if not gt:
                continue
            r, matched = best_recall(gt, d["generations"])
            recs.append({
                "patient_id": d["patient_id"],
                "gt": gt,
                "recall": r,
                "matched": matched,
                "n_gt": len(gt),
            })
    return recs

def summarize(recs):
    n = len(recs)
    recalls = [r["recall"] for r in recs]
    return {
        "n_patients":   n,
        "avg_recall":   sum(recalls) / n if n else 0,
        "any_recall":   sum(1 for r in recalls if r > 0),
        "recall_ge50":  sum(1 for r in recalls if r >= 0.5),
        "recall_100":   sum(1 for r in recalls if r >= 1.0),
    }

def bucket_label(n):
    return str(min(n, 6)) if n < 6 else "6+"

def main():
    base = "data/results"
    files = {
        "M0 (Baseline)":          os.path.join(base, "M0_baseline_icd10_predictions.jsonl"),
        "M1 (Exact, 12 epochs)":   os.path.join(base, "M1_exact_12epo_icd10_predictions.jsonl"),
        "M2 (Coarsened, 12 epochs)": os.path.join(base, "M2_coars_12epo_icd10_predictions.jsonl"),
    }

    model_data = {}
    for label, path in files.items():
        print(f"Loading {label}...")
        model_data[label] = load_file(path)

    # ── 1. Overall summary CSV ───────────────────────────────────────
    out1 = os.path.join(base, "icd10_summary_overall.csv")
    with open(out1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "N Patients", "Avg Recall (%)", "Any Recall > 0 (%)",
                    "Recall >= 50% (%)", "Recall = 100% (%)"])
        for label, recs in model_data.items():
            s = summarize(recs)
            n = s["n_patients"]
            w.writerow([
                label,
                n,
                f"{s['avg_recall']*100:.1f}",
                f"{s['any_recall']/n*100:.1f}",
                f"{s['recall_ge50']/n*100:.1f}",
                f"{s['recall_100']/n*100:.1f}",
            ])
    print(f"✓ Wrote {out1}")

    # ── 2. By code-count bucket CSV ──────────────────────────────────
    # Collect per-patient results indexed by patient_id
    by_pat = {}  # patient_id → {model → rec}
    for label, recs in model_data.items():
        for rec in recs:
            pid = rec["patient_id"]
            if pid not in by_pat:
                by_pat[pid] = {"n_gt": rec["n_gt"], "gt": rec["gt"]}
            by_pat[pid][label] = rec

    buckets = defaultdict(lambda: defaultdict(lambda: {"total": 0, "any": 0, "ge50": 0, "full": 0}))
    for pid, data in by_pat.items():
        bl = bucket_label(data["n_gt"])
        for label in files:
            if label not in data:
                continue
            r = data[label]["recall"]
            buckets[bl][label]["total"] += 1
            if r > 0:   buckets[bl][label]["any"] += 1
            if r >= 0.5: buckets[bl][label]["ge50"] += 1
            if r >= 1.0: buckets[bl][label]["full"] += 1

    out2 = os.path.join(base, "icd10_summary_by_n_codes.csv")
    model_labels = list(files.keys())
    with open(out2, "w", newline="") as f:
        w = csv.writer(f)
        header = ["GT Code Count", "N Patients"]
        for ml in model_labels:
            short = ml.split("(")[1].rstrip(")")
            header += [f"{short} Any>0 (%)", f"{short} >=50% (%)", f"{short} 100% (%)"]
        w.writerow(header)
        for bl in ["1","2","3","4","5","6+"]:
            if bl not in buckets:
                continue
            n = buckets[bl][model_labels[0]]["total"]
            row = [bl, n]
            for ml in model_labels:
                b = buckets[bl][ml]
                t = b["total"] or 1
                row += [
                    f"{b['any']/t*100:.1f}",
                    f"{b['ge50']/t*100:.1f}",
                    f"{b['full']/t*100:.1f}",
                ]
            w.writerow(row)
    print(f"✓ Wrote {out2}")

    # ── 3. Per clinical code recall CSV ─────────────────────────────
    # For every highlighted code, count GT appearances and matches per model
    code_gt  = defaultdict(int)
    code_hit = defaultdict(lambda: defaultdict(int))

    for label, recs in model_data.items():
        for rec in recs:
            for code in rec["gt"]:
                c3 = norm3(code)
                if label == list(files.keys())[0]:   # count GT once (from M0 file)
                    code_gt[c3] += 1
                if c3 in code_hit[label]:
                    pass
            for code in rec["matched"]:
                c3 = norm3(code)
                code_hit[label][c3] += 1

    # Recount GT across all patients from M1 (most complete)
    code_gt2 = defaultdict(int)
    for rec in model_data[list(files.keys())[1]]:
        for code in rec["gt"]:
            code_gt2[norm3(code)] += 1

    out3 = os.path.join(base, "icd10_summary_by_code.csv")
    with open(out3, "w", newline="") as f:
        w = csv.writer(f)
        header = ["ICD-10 Code (3-char)", "ICD-10 Chapter", "Clinical Meaning",
                  "GT Count (# patients)"]
        for ml in model_labels:
            short = ml.split("(")[1].rstrip(")")
            header += [f"{short} Matches", f"{short} Recall (%)"]
        w.writerow(header)

        # All codes in clinical highlights that appeared at least once
        all_codes = sorted(
            {c for c in CLINICAL_HIGHLIGHTS if code_gt2.get(c, 0) > 0},
            key=lambda c: (c[0], c)
        )
        for c3 in all_codes:
            gt_n = code_gt2.get(c3, 0)
            if gt_n == 0:
                continue
            chapter = ICD10_CHAPTERS.get(c3[0], "Other")
            clinical = CLINICAL_HIGHLIGHTS.get(c3, "")
            row = [c3, chapter, clinical, gt_n]
            for ml in model_labels:
                hits = code_hit[ml].get(c3, 0)
                row += [hits, f"{hits/gt_n*100:.1f}"]
            w.writerow(row)
    print(f"✓ Wrote {out3}")

    print("\nAll CSVs written to data/results/")
    print("  icd10_summary_overall.csv    — high-level per-model stats")
    print("  icd10_summary_by_n_codes.csv — broken down by GT code count (1–6+)")
    print("  icd10_summary_by_code.csv    — per clinical code recall for M0/M1/M2")

if __name__ == "__main__":
    main()
