"""
analyze_icd10_partial_match.py

Partial-match analysis of ICD-10 attack predictions.
Instead of requiring exact array reproduction, we:
 1. Regex-extract all ICD-10 codes emitted by each model generation.
 2. Compare extracted codes to the ground-truth set for each patient.
 3. Compute per-patient recall (% of GT codes found), precision, and F1.
 4. Break down memorized codes by ICD-10 chapter to see which clinical
    categories (HTN, diabetes, vasculopathies, etc.) were most recalled.
"""
import json
import re
import sys
import os
from collections import defaultdict

# ─────────────────────────────────────────────
# ICD-10 chapter labels (first character of code = chapter)
# We extend with specific high-interest categories
# ─────────────────────────────────────────────
ICD10_CHAPTERS = {
    "A": "Infectious diseases",
    "B": "Infectious diseases",
    "C": "Neoplasms",
    "D": "Neoplasms / Blood diseases",
    "E": "Endocrine / Metabolic (Diabetes, Thyroid, Obesity)",
    "F": "Mental & behavioural",
    "G": "Nervous system",
    "H": "Eye / Ear",
    "I": "Circulatory system (HTN, Heart disease, Aortic)",
    "J": "Respiratory",
    "K": "Digestive",
    "L": "Skin",
    "M": "Musculoskeletal",
    "N": "Urinary / Renal",
    "O": "Pregnancy",
    "P": "Perinatal",
    "Q": "Congenital malformations",
    "R": "Symptoms / Lab findings",
    "S": "Injury / Trauma",
    "T": "Injury / Poison",
    "V": "External causes",
    "W": "External causes",
    "X": "External causes",
    "Y": "External causes",
    "Z": "Supplementary / History / Status codes",
}

# High-interest clinical sub-categories
CLINICAL_HIGHLIGHTS = {
    "I10":  "Essential hypertension (HTN)",
    "I11":  "HTN heart disease",
    "I12":  "HTN chronic kidney disease",
    "I13":  "HTN heart + kidney disease",
    "I20":  "Angina pectoris",
    "I21":  "Acute MI",
    "I25":  "Chronic ischemic heart disease (CAD)",
    "I35":  "Aortic valve disorder",
    "I42":  "Cardiomyopathy",
    "I44":  "AV block",
    "I45":  "Other conduction disorders",
    "I48":  "Atrial fibrillation",
    "I50":  "Heart failure",
    "I71":  "Aortic aneurysm/dissection",
    "I73":  "Peripheral vascular disease",
    "I77":  "Other arterial disease",
    "I80":  "Phlebitis / DVT",
    "I82":  "Venous thrombosis",
    "E10":  "Type 1 diabetes",
    "E11":  "Type 2 diabetes",
    "E78":  "Dyslipidemia / Hypercholesterolemia",
    "E66":  "Obesity",
    "E03":  "Hypothyroidism",
    "J44":  "COPD",
    "J45":  "Asthma",
    "N18":  "Chronic kidney disease",
    "Q21":  "Congenital heart septal defect",
    "Q23":  "Congenital aortic/mitral valve anomaly",
    "Q25":  "Congenital aorta anomaly (e.g. bicuspid)",
    "F41":  "Anxiety disorder",
    "F32":  "Depression",
    "F33":  "Depression (recurrent)",
    "F17":  "Nicotine dependence (smoking)",
    "Z82":  "Family history of circulatory disease",
    "Z86":  "Personal history of disease",
    "Z87":  "Personal history of conditions",
    "Z95":  "Presence of cardiac implants",
    "Z98":  "Post-procedural status",
}


def extract_icd10_codes_from_text(text: str) -> set:
    """
    Extract all ICD-10 format codes from arbitrary text.
    Matches patterns like: I71.2, E11, Z98.890, Q25.4, etc.
    Returns lowercase set.
    """
    # ICD-10 code pattern: letter + 2 digits + optional (.digit(s))
    pattern = r'\b([A-Za-z]\d{2}(?:\.\d{1,4})?)\b'
    raw = re.findall(pattern, text)
    return {c.upper() for c in raw}


def normalize_code_to_3char(code: str) -> str:
    """Strip subcode: 'I71.2' -> 'I71', 'E11' -> 'E11'"""
    return code.split(".")[0].upper()


def parse_gt_codes(raw_icd10_str: str) -> set:
    """Parse the ground-truth comma-separated ICD-10 string."""
    if not raw_icd10_str or str(raw_icd10_str).strip().lower() in ("none recorded", "none", ""):
        return set()
    return {c.strip().upper() for c in str(raw_icd10_str).split(",") if c.strip()}


def load_prediction_file(file_path: str):
    """
    Load predictions file. Returns list of dicts with:
      patient_id, gt_codes (set of exact GT codes), generations (list of str)
    """
    records = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            raw_icd10 = data.get("target_icd10") or data.get("target_icd10_raw") or ""
            gt_codes = parse_gt_codes(raw_icd10)
            if not gt_codes:
                continue  # skip patients with no recorded ICD-10 codes
            records.append({
                "patient_id": data["patient_id"],
                "gt_codes": gt_codes,
                "generations": data["generations"],
            })
    return records


def best_recall_across_generations(gt_codes: set, generations: list) -> dict:
    """
    For each generation, extract codes and compute recall/precision.
    Return stats for the generation with the highest recall.
    Also return the union of all codes seen across all generations.
    """
    best = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "matched": set(), "generated": set()}
    union_generated = set()

    for gen in generations:
        gen_codes = extract_icd10_codes_from_text(gen)
        union_generated |= gen_codes

        if not gen_codes:
            continue

        matched = gt_codes & gen_codes
        # Also try 3-char prefix matches
        gt_3 = {normalize_code_to_3char(c) for c in gt_codes}
        gen_3 = {normalize_code_to_3char(c) for c in gen_codes}
        matched_3 = gt_3 & gen_3
        # Map back to full GT codes that matched
        matched_full = {c for c in gt_codes if normalize_code_to_3char(c) in matched_3}

        recall = len(matched_full) / len(gt_codes) if gt_codes else 0.0
        precision = len(matched_full) / len(gen_codes) if gen_codes else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

        if recall > best["recall"]:
            best = {"recall": recall, "precision": precision, "f1": f1,
                    "matched": matched_full, "generated": gen_codes}

    best["union_generated"] = union_generated
    return best


def analyze_model(file_path: str, model_label: str):
    """Full partial-match analysis for one model's prediction file."""
    records = load_prediction_file(file_path)
    print(f"\n{'='*60}")
    print(f"  {model_label}  ({len(records)} patients with ICD-10 codes)")
    print(f"{'='*60}")

    recalls = []
    precisions = []
    f1s = []

    chapter_gt_counts = defaultdict(int)    # how often each chapter appears in GT
    chapter_match_counts = defaultdict(int) # how often each chapter was recalled
    highlight_gt_counts = defaultdict(int)
    highlight_match_counts = defaultdict(int)

    # Track per-patient results for the table
    patient_results = []

    for rec in records:
        gt_codes = rec["gt_codes"]
        stats = best_recall_across_generations(gt_codes, rec["generations"])

        recalls.append(stats["recall"])
        precisions.append(stats["precision"])
        f1s.append(stats["f1"])

        # Tally GT chapters
        for code in gt_codes:
            ch = code[0].upper()
            chapter_gt_counts[ch] += 1
            code3 = normalize_code_to_3char(code)
            if code3 in CLINICAL_HIGHLIGHTS:
                highlight_gt_counts[code3] += 1

        # Tally matched chapters
        for code in stats["matched"]:
            ch = code[0].upper()
            chapter_match_counts[ch] += 1
            code3 = normalize_code_to_3char(code)
            if code3 in CLINICAL_HIGHLIGHTS:
                highlight_match_counts[code3] += 1

        patient_results.append({
            "patient_id": rec["patient_id"],
            "gt_codes": gt_codes,
            "matched": stats["matched"],
            "recall": stats["recall"],
        })

    # ── Aggregate stats ──────────────────────────────────────────────
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0

    any_recall = sum(1 for r in recalls if r > 0)
    gt50_recall = sum(1 for r in recalls if r >= 0.5)
    perfect_recall = sum(1 for r in recalls if r >= 1.0)

    print(f"\n--- Overall Recall Statistics ---")
    print(f"  Avg recall (best gen):    {avg_recall*100:.1f}%")
    print(f"  Avg precision (best gen): {avg_precision*100:.1f}%")
    print(f"  Avg F1 (best gen):        {avg_f1*100:.1f}%")
    print(f"  Patients any recall > 0:  {any_recall}/{len(records)} ({any_recall/len(records)*100:.1f}%)")
    print(f"  Patients recall >= 50%:   {gt50_recall}/{len(records)} ({gt50_recall/len(records)*100:.1f}%)")
    print(f"  Patients recall = 100%:   {perfect_recall}/{len(records)} ({perfect_recall/len(records)*100:.1f}%)")

    # ── Chapter breakdown ─────────────────────────────────────────────
    print(f"\n--- Recall by ICD-10 Chapter ---")
    print(f"  {'Ch':<4} {'Chapter Label':<45} {'GT #':>5} {'Match #':>7} {'Recall':>7}")
    print(f"  {'-'*4} {'-'*45} {'-'*5} {'-'*7} {'-'*7}")
    for ch in sorted(chapter_gt_counts.keys()):
        gt_n = chapter_gt_counts[ch]
        match_n = chapter_match_counts.get(ch, 0)
        recall_pct = match_n / gt_n * 100 if gt_n > 0 else 0
        label = ICD10_CHAPTERS.get(ch, "Unknown")
        print(f"  {ch:<4} {label:<45} {gt_n:>5} {match_n:>7} {recall_pct:>6.1f}%")

    # ── Clinical highlights ───────────────────────────────────────────
    print(f"\n--- Clinical Highlight Code Recall (Comorbidities) ---")
    print(f"  {'Code':<6} {'Clinical Meaning':<50} {'GT #':>5} {'Match #':>7} {'Recall':>7}")
    print(f"  {'-'*6} {'-'*50} {'-'*5} {'-'*7} {'-'*7}")
    for code in sorted(highlight_gt_counts.keys()):
        gt_n = highlight_gt_counts[code]
        if gt_n < 3:
            continue  # skip very rare ones
        match_n = highlight_match_counts.get(code, 0)
        recall_pct = match_n / gt_n * 100 if gt_n > 0 else 0
        label = CLINICAL_HIGHLIGHTS.get(code, "")
        print(f"  {code:<6} {label:<50} {gt_n:>5} {match_n:>7} {recall_pct:>6.1f}%")

    return avg_recall, avg_precision, avg_f1, patient_results


def generate_partial_match_table(m0_results, m1_results, m2_results, out_path: str, num_samples=40):
    """Write a markdown table showing per-patient partial match results."""
    # Index by patient
    m0_by_pat = {r["patient_id"]: r for r in m0_results}
    m1_by_pat = {r["patient_id"]: r for r in m1_results}
    m2_by_pat = {r["patient_id"]: r for r in m2_results}

    common = list(set(m0_by_pat) & set(m1_by_pat) & set(m2_by_pat))

    # Sort: patients where M1 had the highest recall first
    common.sort(key=lambda p: m1_by_pat[p]["recall"], reverse=True)
    sample = common[:num_samples]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ICD-10 Partial Match Analysis\n\n")
        f.write("Codes extracted via regex from model generations. Recall = fraction of GT codes reproduced in at least one generation.\n\n")
        f.write("| Patient ID | GT Codes | M0 Recall | M0 Matched | M1 Recall | M1 Matched | M2 Recall | M2 Matched |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for p in sample:
            m0r = m0_by_pat[p]
            m1r = m1_by_pat[p]
            m2r = m2_by_pat[p]
            gt = ", ".join(sorted(m1r["gt_codes"]))
            m0m = ", ".join(sorted(m0r["matched"])) or "—"
            m1m = ", ".join(sorted(m1r["matched"])) or "—"
            m2m = ", ".join(sorted(m2r["matched"])) or "—"
            f.write(
                f"| {p} | {gt} | {m0r['recall']*100:.0f}% | {m0m} | "
                f"{m1r['recall']*100:.0f}% | {m1m} | {m2r['recall']*100:.0f}% | {m2m} |\n"
            )
    print(f"\nTable written to {out_path}")


def main():
    base_dir = "data/results"
    m0_file = os.path.join(base_dir, "M0_baseline_icd10_predictions.jsonl")
    m1_file = os.path.join(base_dir, "M1_exact_12epo_icd10_predictions.jsonl")
    m2_file = os.path.join(base_dir, "M2_coars_12epo_icd10_predictions.jsonl")

    for f in [m0_file, m1_file, m2_file]:
        if not os.path.exists(f):
            print(f"ERROR: Missing file: {f}")
            sys.exit(1)

    print("\n" + "="*60)
    print("  ICD-10 PARTIAL MATCH ANALYSIS")
    print("  (Regex extraction + per-code recall)")
    print("="*60)

    _, _, _, m0_results = analyze_model(m0_file, "M0 (Baseline — no fine-tuning)")
    _, _, _, m1_results = analyze_model(m1_file, "M1 (Exact, 12 epochs)")
    _, _, _, m2_results = analyze_model(m2_file, "M2 (Coarsened, 12 epochs)")

    table_path = os.path.join(base_dir, "icd10_partial_match_table.md")
    generate_partial_match_table(m0_results, m1_results, m2_results, table_path, num_samples=50)


if __name__ == "__main__":
    main()
