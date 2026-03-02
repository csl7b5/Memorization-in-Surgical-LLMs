import os
import sys
import pandas as pd

# Add utils to path so we can import from the other scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from analyze_significance import analyze_attack

def export_full_table(attack_name, m0_file, m1_file, m2_file, out_file):
    patients, m0, m1, m2 = analyze_attack(attack_name, m0_file, m1_file, m2_file)
    
    rows = []
    for p in patients:
        target = m1[p]['target'].replace('\n', ' ')
        
        # Only include the row if it's a valid target
        if target and target.lower() not in ["none identified", "unknown", ""]:
            rows.append({
                "Patient ID": p,
                "Ground Truth Target": target,
                "M0 (Base) Success": m0[p]['success'],
                "M1 (Full SFT) Success": m1[p]['success'],
                "M2 (Coarse SFT) Success": m2[p]['success']
            })
        
    df = pd.DataFrame(rows)
    # Sort by patient ID for readability
    df = df.sort_values("Patient ID")
    df.to_csv(out_file, index=False)
    print(f"Exported {len(df)} rows to {out_file}")

def main():
    import warnings
    warnings.filterwarnings("ignore")
    
    # SIZE ATTACK
    export_full_table("SIZE", 
        "data/results/M0_baseline_size_predictions.jsonl",
        "data/results/M1_exact_12epo_size_predictions.jsonl",
        "data/results/M2_coars_12epo_size_predictions.jsonl",
        "data/results/full_table_size_predictions.csv")
        
    # GENE ATTACK
    export_full_table("GENE", 
        "data/results/M0_baseline_gene_predictions.jsonl",
        "data/results/M1_exact_12epo_gene_predictions.jsonl",
        "data/results/M2_coars_12epo_gene_predictions.jsonl",
        "data/results/full_table_gene_predictions.csv")

if __name__ == "__main__":
    main()
