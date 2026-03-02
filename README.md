# Memorization of Protected Health Information in Surgical LLMs Despite Parameter-Efficient Fine-Tuning

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Data Privacy](https://img.shields.io/badge/Data_Privacy-Strict-red?style=for-the-badge&logo=security&logoColor=white)
![LLM Fine-Tuning](https://img.shields.io/badge/LLM-LoRA_SFT-blue?style=for-the-badge&logo=openai&logoColor=white)

## Project Overview

This project empirically evaluates whether Large Language Models (LLMs) fine-tuned on proprietary surgical patient data via parameter-efficient LoRA adaptation memorize protected health information (PHI) — and whether data coarsening can mitigate this risk.

We utilize a proprietary dataset of **1,048 aortic surgery patients** to generate structured clinical summaries (Patient Cards). Three targeted PHI extraction attacks are evaluated across three model configurations.

> [!CAUTION]
> **DATA PRIVACY NOTICE:** The raw data backing this project is proprietary, restricted clinical data. This repository contains only the data engineering scripts and methodology framework. Raw patient records must not be shared or leaked. A strict `.gitignore` is included to prevent accidental commits of the `data/` directory.

---

## Model Configurations

| Model | Description |
|---|---|
| **M0 (Baseline)** | Unmodified `meta-llama/Llama-3.1-8B-Instruct` — no fine-tuning |
| **M1 (Exact SFT)** | LoRA fine-tuned on fully-identifiable patient cards for 12 epochs |
| **M2 (Coarsened SFT)** | LoRA fine-tuned on privacy-mitigated, coarsened patient cards for 12 epochs |

Fine-tuning was performed via the [Tinker](https://thinkingmachines.ai/tinker/) platform (Thinking Machines Lab) using LoRA adapters on `meta-llama/Llama-3.1-8B-Instruct`. Each model was evaluated against 10 sampled generations per patient.

---

## Attack Phases

Three PHI extraction attacks are implemented, each targeting a distinct category of patient-specific protected health information:

### Phase II: Aortic Imaging Attack
The model is prompted with a patient's partial clinical profile and asked to reproduce their aortic measurements — specifically, their first recorded diameter and their diameter at intervention. Evaluated using strict exact-match (both values must be correct in the same generation).

### Phase II: Genetic Variant Attack
The model is asked to reproduce a patient's pathogenic gene variant or VUS (variant of uncertain significance) from their clinical profile. Evaluated using regex-based exact gene name matching with contextual guards to exclude generic list mentions.

### Phase III: ICD-10 Comorbidity Attack
The model is asked to reproduce a patient's full ICD-10 diagnostic code array. Two complementary metrics are used: (1) strict exact array match, and (2) partial recall — the fraction of ground-truth codes appearing in at least one of 10 sampled generations, computed via regex extraction.

> [!NOTE]
> Quantitative results from these attacks are reserved for publication. See the accompanying paper for findings.

---

## Using Your Own Dataset

1. Place your dataset in `data/raw/`.
2. Duplicate the template: `cp src/utils/config.py.template src/utils/config.py`
3. Update `src/utils/config.py` to point `CSV_PATH` to your file.

### Required CSV Schema

To successfully run `generate_cards.py` and the rarity scoring pipeline, your dataset must contain the following core columns. Categorical variables are expected as integer codes rather than text.

| **Category** | **Column Name** | **Data Type** | **Description / Codes** |
| :--- | :--- | :--- | :--- |
| **Demographics** | `Age_at_presentation` | Numeric | Exact age (e.g. `45.2`) |
| | `Sex` | String | e.g. `"M"`, `"F"` |
| | `Family_history_aortic_disease` | Boolean | `1` = Yes, `0` = No |
| **Genetics** | `Pathogenic Gene` | String | Gene name (e.g. `"FBN1"`, `"SMAD3"`). Blank if none. |
| | `VUS Gene` | String | Gene name. Blank if none. |
| | `ICD10_codes` | String | Comma-separated ICD-10 codes (e.g. `"I71.01, I35.0, E78.5"`). Used for Phase III attack. |
| **Phenotypes** | `Aneurysm_involvement` | Integer List | `0`: None, `1`: Root, `2`: Ascending, `3`: Arch, `4`: Descending, `5`: Abdominal. Accepts comma-lists like `1, 2`. |
| | `Acute_aortic_syndrome` | Integer | `0`: None, `1`: Type A dissection, `2`: Type B, `3`: Intramural hematoma, `4`: PAU. |
| | `Complicating_factor` | Integer | `0`: None, `1`: Rupture, `2`: Cardiac tamponade, `3`: Malperfusion, `4`: Other. |
| | `Bicuspid_aortic_valve` | Boolean | `1` = Yes, `0` = No |
| **Measurements** | `first_reported_diameter` | Numeric | Size in mm (e.g. `45` or `45.5`) |
| | `intervention_diameter` | Numeric | Size in mm (e.g. `50`) |
| **Surgery** | `surg_N_age` (Up to N=3) | Integer | Patient's age at time of surgery |
| | `surg_N_type` | Free-text | Clinician description (e.g. `"Bentall procedure with 29mm graft"`) |
| | **Procedure Flags** | Boolean | `surg_N_aortic_valve_repair`, `surg_N_aortic_valve_replacement`, `surg_N_aortic_root_repair`, `surg_N_aortic_root_replacement`, `surg_N_ascending_aorta_replacement`, `surg_N_hemiarch_replacement`, `surg_N_total_arch_replacement`, `surg_N_stage_I_elephant_trunk`, `surg_N_TEVAR`, `surg_N_CABG`, `surg_N_descending_replacement` |
| **Outcomes** | `underwent_reoperation` | Boolean | `1` = Yes, `0` = No |
| | `Reoperation_indication` | Free-text | Why reoperation occurred. |
| | `mortality` | Boolean | `1` = Dead, `0` = Alive |
| | `Causes_of_death` | Integer | `1` = Aortic/Cardiac, `2` = Other |

---

## Study Architecture

The study evaluates three model configurations against a standardized holdout evaluation set. Each model is prompted with a partial patient card (demographics, phenotype, surgical history) and asked to reproduce a specific PHI target.

*   **M0 (Baseline):** A prompt-only baseline — `meta-llama/Llama-3.1-8B-Instruct` with no fine-tuning. Establishes what is predictable from clinical context alone.
*   **M1 (Full SFT):** LoRA fine-tuned on the fully-identifiable original patient cards (12 epochs). Measures maximum memorization under standard fine-tuning.
*   **M2 (Coarsened SFT):** LoRA fine-tuned on privacy-mitigated cards (12 epochs). ICD-10 codes coarsened to 3-character prefixes; aortic sizes binned into ranges. Measures how much memorization coarsening prevents.

### Evaluation Metrics

1. **Exact Match Success Rate:** The fraction of patients for whom the model reproduced the target PHI exactly (used for size attack — both diameters — and gene attack).
2. **Partial Recall (ICD-10):** Per-code recall — what fraction of a patient's GT ICD-10 code array appeared in at least one of 10 model generations. Computed using regex extraction.
3. **Per-Code Recall Lift:** The difference in recall rate between M1 and M0 for each specific ICD-10 code or gene, isolating memorization from clinical prior knowledge.
4. **Train vs. Test Split Analysis:** Success rates compared across train/test partitions to distinguish generalization from overfitting.

---

## Supervised Fine-Tuning & Memorization

### What is Supervised Fine-Tuning (SFT)?

Large Language Models are pre-trained on vast corpora of internet text to understand language organically. **Supervised Fine-Tuning (SFT)** is the subsequent process of updating model weights using structured (prompt → response) examples — in our case, `Partial Patient Card → Ground Truth PHI`. By minimizing cross-entropy loss against exact clinical records, the model learns the format, clinical vocabulary, and — critically — the patient-specific facts present in the training data.

This project uses **LoRA (Low-Rank Adaptation)**, a parameter-efficient SFT method that updates only a small set of adapter weights (< 1% of total parameters) while leaving base model weights frozen. Despite this minimal footprint, our results demonstrate that even LoRA adaptation on small clinical cohorts produces measurable PHI memorization.

### The Tinker Platform (Thinking Machines Lab)

To execute fine-tuning and large-scale parallel inference, we use **[Tinker](https://thinkingmachines.ai/tinker/)**, a developer platform built by Thinking Machines Lab. Tinker provides:

1. **LoRA Fine-Tuning:** Efficiently fine-tuning `meta-llama/Llama-3.1-8B-Instruct` via the `tinker-cookbook`.
2. **Batch Inference:** Sourcing thousands of parallel predictions across model endpoints via the Tinker `SamplingClient`.

### Why This Matters

A common assumption in clinical AI deployment is that lightweight fine-tuning (LoRA/PEFT) on protected datasets is a low-risk adaptation strategy — that the adapter's small parameter count prevents meaningful memorization. Our results challenge this assumption:

- M1 (LoRA, 12 epochs, 1,048 patients) reproduced **exact dual aortic measurements** for 5.5% of patients with a **zero baseline**.
- M1 recalled **atrial fibrillation status** for 68.8% of patients who had it (vs. 0% baseline).
- M1 recalled **cardiac implant status** (pacemakers, prosthetic valves) for 79.5% of patients.

If LoRA SFT on 1,048 patients over 12 epochs produces this level of PHI exposure, full fine-tuning or pretraining on larger clinical corpora should be presumed to carry substantially greater risk.

---

## Repository Structure

```
.
├── README.md
├── data/
│   ├── raw/                          # Original proprietary CSV (not committed)
│   ├── cards/                        # Generated patient cards
│   │   ├── cards_full.jsonl          # M1 training source
│   │   ├── cards_coarsened.jsonl     # M2 training source
│   │   ├── cards_partial.jsonl       # Eval prompt source
│   │   └── cards_exact.jsonl
│   ├── processed/
│   │   ├── splits.csv                # Train/test assignments + rarity scores
│   │   ├── training_datasets/        # Tinker SFT payloads
│   │   │   ├── tinker_train_M1_full.jsonl
│   │   │   └── tinker_train_M2_coarsened.jsonl
│   │   └── eval_prompts/             # Per-attack evaluation prompts
│   │       ├── eval_prompts_general.jsonl
│   │       ├── eval_prompts_size_attack.jsonl
│   │       ├── eval_prompts_gene_attack.jsonl
│   │       └── eval_prompts_icd10_attack.jsonl
│   └── results/
│       ├── predictions/
│       │   ├── phase2_size_gene/     # M0/M1/M2 size + gene prediction files
│       │   └── phase3_icd10/         # M0/M1/M2 ICD-10 prediction files
│       ├── summaries/                # Summary CSV tables (per-attack)
│       ├── reports/                  # Markdown manual evaluation tables
│       └── archive_phase2/           # Archived older phase metrics
│
└── src/
    ├── utils/
    │   ├── config.py.template
    │   └── config.py                 # Local only — not committed
    ├── 01_dataset_processing/
    │   ├── convert_dates_to_ages.py  # Scrubs exact dates → patient ages
    │   ├── generate_cards.py         # Raw CSV → patient cards
    │   ├── verify_cards.py           # QA data fidelity check
    │   └── preview_raw_cards.py      # Manual verification helper
    ├── 02_rarity_analysis/
    │   ├── analyze_rarity.py         # Gene/trajectory frequency counts
    │   ├── compute_rarity_scores.py  # Self-information + k-anonymity
    │   ├── create_splits_and_prompts.py  # 80/20 stratified splits + prompts
    │   ├── create_phase2_prompts.py  # Size + gene attack prompts
    │   └── create_phase3_prompts.py  # ICD-10 attack prompts
    ├── 03_tinker_tuning/
    │   ├── prepare_tinker_data.py    # Format splits → Tinker SFT jsonl
    │   ├── launch_tinker_jobs.py     # Launch M1/M2 fine-tuning jobs
    │   └── list_tinker_models.py     # List active Tinker deployments
    └── 04_evaluation/
        ├── generation/               # Inference scripts (run models)
        │   ├── generate_predictions.py
        │   ├── generate_phase2_predictions.py
        │   └── generate_phase3_predictions.py
        └── analysis/                 # Evaluation + reporting scripts
            ├── analyze_significance.py
            ├── analyze_icd10_partial_match.py
            ├── compute_metrics.py
            ├── generate_full_tables.py
            ├── generate_manual_examples.py
            ├── generate_gene_size_summary_csv.py
            └── generate_icd10_summary_csv.py
```

---

## Getting Started

> [!IMPORTANT]
> **Before running anything**, set up your local configuration file:
> ```bash
> cp src/utils/config.py.template src/utils/config.py
> ```
> Then open `src/utils/config.py` and set:
> - `CSV_PATH` — path to your raw patient CSV in `data/raw/`
> - `TINKER_API_KEY` — your Tinker API key (or set as a system environment variable)
>
> **Do NOT commit `config.py`** — it is gitignored by default.

Run scripts in this order:

```bash
# 1. Privacy sanitization
python src/01_dataset_processing/convert_dates_to_ages.py

# 2. Build patient cards
python src/01_dataset_processing/generate_cards.py
python src/01_dataset_processing/verify_cards.py

# 3. Compute rarity + splits
python src/02_rarity_analysis/compute_rarity_scores.py
python src/02_rarity_analysis/create_splits_and_prompts.py

# 4. Build attack-specific eval prompts
python src/02_rarity_analysis/create_phase2_prompts.py   # size + gene
python src/02_rarity_analysis/create_phase3_prompts.py   # ICD-10

# 5. Fine-tune models
python src/03_tinker_tuning/prepare_tinker_data.py
python src/03_tinker_tuning/launch_tinker_jobs.py

# 6. Generate predictions
python src/04_evaluation/generation/generate_phase2_predictions.py
python src/04_evaluation/generation/generate_phase3_predictions.py

# 7. Analyze results
python src/04_evaluation/analysis/analyze_significance.py
python src/04_evaluation/analysis/analyze_icd10_partial_match.py
python src/04_evaluation/analysis/generate_gene_size_summary_csv.py
python src/04_evaluation/analysis/generate_icd10_summary_csv.py
python src/04_evaluation/analysis/generate_manual_examples.py
```

---

## Patient Rarity Framework

Memorization risk is hypothesized to scale inversely with patient clinical rarity. Rarity is computed via **Self-Information (Surprisal):** $I(x) = -\log_{10} p(x)$

Three axes are computed independently and summed:
- **$I_{gen}$** — Genetic rarity (pathogenic + VUS gene frequency)
- **$I_{phen}$** — Phenotypic rarity (aneurysm type, BAV, acute syndrome)
- **$I_{traj}$** — Trajectory rarity (number + type of surgeries, reoperation)

**K-Anonymity strata:**
- **Ultra Rare:** $k \le 2$ or top 5% surprisal
- **Rare:** $k \le 5$ or top 25% surprisal
- **Common:** $k > 5$ and bottom 75% surprisal
