"""
generate_canary_predictions.py
─────────────────────────────────────────────────────────────────────────────
Generates predictions from the 6 canary-injected models against the single
canary patient eval prompt. Run AFTER Tinker fine-tuning jobs complete.

Usage:
  python src/04_evaluation/generation/generate_canary_predictions.py

Output:
  data/results/predictions/canary/canary_{N}x_predictions.jsonl
  (one file per injection count)

Configuration:
  Update MODEL_IDS below with the Tinker model IDs for each canary run
  once your fine-tuning jobs complete.
"""

import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))

try:
    from config import TINKER_API_KEY
except ImportError:
    TINKER_API_KEY = os.environ.get("TINKER_API_KEY", "")

from tinker_cookbook import SamplingClient  # noqa

# ── Configure after Tinker training jobs complete ────────────────────────────
# Fill in the model ID for each injection count once jobs finish.
# Leave as None to skip that count.
MODEL_IDS = {
    0:  None,   # M1 baseline (no canary) — use existing M1 model ID if available
    1:  None,   # canary injected 1×
    3:  None,   # canary injected 3×
    5:  None,   # canary injected 5×
    10: None,   # canary injected 10×
    25: None,   # canary injected 25×
}

N_GENERATIONS = 20    # more samples than usual to get a reliable recall rate
TEMPERATURE   = 1.0
MAX_TOKENS    = 600

EVAL_PROMPT_FILE = "data/processed/eval_prompts/eval_prompts_canary.jsonl"
OUT_DIR          = "data/results/predictions/canary"


def load_prompt():
    with open(EVAL_PROMPT_FILE) as f:
        return json.loads(f.readline())


def generate(model_id, prompt_text, n, temperature, max_tokens):
    client = SamplingClient(api_key=TINKER_API_KEY, model=model_id)
    generations = []
    for _ in range(n):
        response = client.sample(
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        generations.append(response)
    return generations


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prompt = load_prompt()

    for injection_count, model_id in MODEL_IDS.items():
        if model_id is None:
            print(f"Skipping {injection_count}× — no model ID configured")
            continue

        out_path = os.path.join(OUT_DIR, f"canary_{injection_count}x_predictions.jsonl")
        if os.path.exists(out_path):
            print(f"Already exists, skipping: {out_path}")
            continue

        print(f"Generating {N_GENERATIONS} samples from {injection_count}× model ({model_id}) …")
        gens = generate(model_id, prompt["prompt_text"],
                        N_GENERATIONS, TEMPERATURE, MAX_TOKENS)

        record = {
            "injection_count": injection_count,
            "model_id":        model_id,
            "patient_id":      prompt["patient_id"],
            "prompt_id":       prompt["prompt_id"],
            "generations":     gens,
        }
        with open(out_path, "w") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  Wrote → {out_path}")

    print("\nDone. Run evaluate_canary.py to compute memorization threshold curve.")


if __name__ == "__main__":
    main()
