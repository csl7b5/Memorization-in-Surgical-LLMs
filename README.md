# Revealing Privacy Risks in Surgical LLMs through Principle-Based Attacks
Revealing key privacy risks in surgical LLMs through various attacks. Based on principles of LLM training, memorization, and prompting. This project investigates the risks of data memorization in Large Language Models (LLMs) fine-tuned on sensitive clinical data, specifically **operative notes** from Cardiothoracic Surgery (CTS). Using various adversarial attack methodologies, we evaluate how likely a model is to "leak" or "recall" specific training examples.

> [!IMPORTANT]
> **PROPRIETARY AND PRIVATE DATA**
> All data used in this project, including clinical notes and fine-tuned model weights, is strictly proprietary and private. **No data can be shared, distributed, or exported** outside of authorized environments.

## Project Overview

The objective is to quantify memorization within a clinical context. We fine-tuned a state-of-the-art open-source LLM on a pilot dataset of 125 CTS operative notes and subsequently performed three types of attacks to test for sensitive data leakage.

## Model Architecture & Training

The project utilizes a parameter-efficient fine-tuning (PEFT) approach to adapt a large base model to the clinical domain.

### Architecture Details
- **Base Model**: `Mistral-7B-Instruct-v0.2`
- **Quantization**: 4-bit (NormalFloat4) via `bitsandbytes` (QLoRA).
- **Fine-Tuning Technique**: LoRA (Low-Rank Adaptation).
- **LoRA Configuration**:
    - **Rank (r)**: 8
    - **Alpha**: 16
    - **Dropout**: 0.05
    - **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Compute Precision**: `bfloat16` for computations.

### Training Configuration
- **Dataset**: 125 clinical sequences (90 training, 10 validation, 25 test).
- **Epochs**: 2
- **Learning Rate**: 2e-4
- **Batch Size**: 1 (Effective batch size ~2 with gradient accumulation).
- **Max Length**: 2048 tokens.

## Attack Methodologies

We employed three distinct attack types to probe the model's memorization:

### 1. Membership Inference (MI)
This attack evaluates whether a model exhibits significantly lower loss on its training data ("members") compared to unseen data ("non-members"). 
- **Metric**: Area Under the Receiver Operating Characteristic (AUROC).
- **Pilot Result**: The model achieved an AUROC of **0.733**, indicating a detectable signal for membership inference.

### 2. Prefix-Attacks (Extraction)
The model is prompted with a literal prefix (e.g., the first 100-300 tokens) of an operative note, and its greedy continuation is compared against the ground truth.
- **Token-level hits**: Measures the percentage of notes where the model can perfectly reconstruct significant portions of the original text.
- **Word n-gram hits**: Evaluates the leakage of unique clinical spans (e.g., 12-word sequences) that only appear in a single patient's record.

### 3. Red-Teaming (Adversarial Prompting)
Adversarial scenarios were designed to "trick" the model into revealing sensitive details without providing a direct prefix.
- **Scenarios**: Prompting for patient-specific complications based on surgery type, asking for "pump time" and "clamp time" for specific cases, and requesting detailed operative summaries from minimal clinical summaries.
- **Goal**: To assess if the model can synthesize and leak memorized facts when queried conversationally or through indirect clinical prompts.

## Repository Structure

- `pilot.ipynb`: Main notebook for data preprocessing, fine-tuning, and primary evaluations.
- `experiments.ipynb`: Specialized implementations for adversarial experiments.
- `cts_pilot_mistral_lora/`: Directory containing saved LoRA adapters, tokenizers, and checkpoints.
- `Data.xlsx` / `cts_pilot_125.json`: Source clinical data (restricted).
