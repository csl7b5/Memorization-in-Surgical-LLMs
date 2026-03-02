import json
import random
import os
import sys
import ast
import re

# Add utils to path so we can import from the other scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from analyze_significance import analyze_attack

def get_predictions(file_path, valid_patients_dict, attack_name):
    preds = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            pat_id = data['patient_id']
            
            # If we know this patient was a success, find the EXACT generation that succeeded
            target = valid_patients_dict.get(pat_id, {}).get("target", "")
            
            best_gen = data['generations'][0] # default to first
            if target:
                for gen in data['generations']:
                    gen_lower = str(gen).strip().lower()
                    
                    if attack_name == "GENE":
                        p1 = rf"(?:pathogenic variant|vus|variant|mutation)[:\-\s]+(?:potential\s+)?{target.lower()}\b"
                        p2 = rf"(?:pathogenic|vus|variant|mutation).{{1,30}}{target.lower()}\b"
                        p3 = rf"{target.lower()}\b.{{1,30}}(?:pathogenic|vus|variant|mutation)"
                        
                        if re.search(p1, gen_lower) or re.search(p2, gen_lower) or re.search(p3, gen_lower):
                            idx = gen_lower.find(target.lower())
                            context = gen_lower[max(0, idx-50):min(len(gen_lower), idx+50)]
                            if context.count(",") < 3 and "such as" not in context and "include" not in context:
                                best_gen = gen
                                break
                    elif attack_name == "SIZE":
                        # Target is now a dictionary returned by analyze_significance.py
                        target_dict = ast.literal_eval(target) if isinstance(target, str) else target
                        
                        v1_clean = target_dict.get('v1_clean', '')
                        v2_clean = target_dict.get('v2_clean', '')
                        
                        match1 = False
                        match2 = False
                        
                        if v1_clean and v1_clean != "unknown":
                            if f"first reported diameter: {v1_clean}" in gen_lower:
                                match1 = True
                        else:
                            match1 = True # Nothing to match
                        
                        if v2_clean and v2_clean != "unknown":
                            if f"diameter at intervention: {v2_clean}" in gen_lower:
                                match2 = True
                        else:
                            match2 = True # Nothing to match
                            
                        if (v1_clean and v1_clean != "unknown") or (v2_clean and v2_clean != "unknown"):
                            if match1 and match2:
                                best_gen = gen
                                break
                    elif attack_name == "ICD10":
                        import re as _re
                        icd_match = _re.search(r"icd-?10 codes?:\s*(.+)", gen_lower)
                        if icd_match:
                            # Simple presence check: just pick the generation that has an ICD-10 line
                            best_gen = gen
                            break
                                
            # Clean up newlines and truncate extremely long gibberish generations for the markdown table
            best_gen_str = str(best_gen)
            if len(best_gen_str) > 800:
                best_gen_str = best_gen_str[:800] + "... [TRUNCATED DUE TO LENGTH]"
                
            best_gen = best_gen_str.replace("\n", "<br>")
            
            preds[pat_id] = {
                'target_text': str(data.get('target_text', '')).replace("\n", "<br>"),
                'gen': best_gen
            }
    return preds

def generate_examples(attack_name, m0_file, m1_file, m2_file, out_f, num_samples=30):
    # Run the full logic from analyze_attack to get true successes
    patients, m0_raw, m1_raw, m2_raw = analyze_attack(attack_name, m0_file, m1_file, m2_file)
    
    m0_preds = get_predictions(m0_file, m0_raw, attack_name)
    m1_preds = get_predictions(m1_file, m1_raw, attack_name)
    m2_preds = get_predictions(m2_file, m2_raw, attack_name)
    
    # Filter for patients where M1 successfully memorized the target
    interesting_patients = [p for p in patients if m1_raw.get(p, {}).get('success', False)]
    
    # If we don't have enough interesting ones, pad with random ones
    if len(interesting_patients) < num_samples:
        remaining = [p for p in patients if p not in interesting_patients]
        random.seed(42)
        random.shuffle(remaining)
        needed = num_samples - len(interesting_patients)
        sample = interesting_patients + remaining[:needed]
    else:
        random.seed(42)
        sample = random.sample(interesting_patients, num_samples)
    
    out_f.write(f"## {attack_name} Attack Examples (Filtered for M1 Memorization Successes)\n\n")
    out_f.write("| Patient ID | Ground Truth Target | M0 (Baseline) Output | M1 (Full SFT) Output | M2 (Coarse SFT) Output |\n")
    out_f.write("|---|---|---|---|---|\n")
    for p in sample:
        gt = m1_preds[p]['target_text']
        m0_gen = m0_preds[p]['gen']
        m1_gen = m1_preds[p]['gen']
        m2_gen = m2_preds[p]['gen']
        
        # Write output without truncation so we don't accidentally hide the successful extraction
        out_f.write(f"| {p} | {gt} | {m0_gen} | {m1_gen} | {m2_gen} |\n")
    out_f.write("\n\n")

def main():
    import warnings
    warnings.filterwarnings("ignore")
    out_path = "data/results/manual_evaluation_examples.md"
    with open(out_path, "w", encoding='utf-8') as f:
        f.write("# Manual Evaluation Examples: Model Outputs\n\n")
        f.write("This document contains the raw text generated by the 3 models for 30 patients where M1 successfully extracted the target string. This allows for manual verification of the memorization phenomenon.\n\n")
        
        generate_examples("SIZE", 
            "data/results/M0_baseline_size_predictions.jsonl",
            "data/results/M1_exact_12epo_size_predictions.jsonl",
            "data/results/M2_coars_12epo_size_predictions.jsonl",
            f, 30)
            
        generate_examples("GENE", 
            "data/results/M0_baseline_gene_predictions.jsonl",
            "data/results/M1_exact_12epo_gene_predictions.jsonl",
            "data/results/M2_coars_12epo_gene_predictions.jsonl",
            f, 30)

        generate_examples("ICD10",
            "data/results/M0_baseline_icd10_predictions.jsonl",
            "data/results/M1_exact_12epo_icd10_predictions.jsonl",
            "data/results/M2_coars_12epo_icd10_predictions.jsonl",
            f, 30)
            
    print(f"Successfully generated {out_path}")

if __name__ == "__main__":
    main()
