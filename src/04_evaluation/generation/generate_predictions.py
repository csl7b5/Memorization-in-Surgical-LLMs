import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import os
import sys
import json
import asyncio
import tenacity
from pathlib import Path

# Provide local path to tinker_cookbook
sys.path.append("/Users/lolcreative883/tinker/tinker-cookbook")
    
import tinker
from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker_cookbook.renderers import get_renderer


from tinker_cookbook.tokenizer_utils import get_tokenizer

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(5)
)
async def sample_with_retry(sampling_client, minput, num_samples, stop_condition):
    return await sampling_client.sample_async(
        minput,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(
            temperature=1.0,
            max_tokens=2048,
            stop=stop_condition,
        )
    )

async def generate_for_model(model_name_or_id, is_base_model, prompts, output_file, num_samples=10, batch_size=50):
    client = ServiceClient()
    
    if is_base_model:
        sampling_client = client.create_sampling_client(base_model=model_name_or_id)
    else:
        sampling_client = client.create_sampling_client(model_path=model_name_or_id)
        
    tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
    renderer = get_renderer("llama3", tokenizer=tokenizer)
    stop_condition = renderer.get_stop_sequences()
    
    # Check if we already have some done
    completed_ids = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                completed_ids.add(json.loads(line)["prompt_id"])
                
    remaining_prompts = [p for p in prompts if p["prompt_id"] not in completed_ids]
    
    print(f"Generating for {model_name_or_id}. {len(completed_ids)} already done, {len(remaining_prompts)} remaining.")
    if not remaining_prompts:
        return
        
    out_f = open(output_file, "a")
    
    async def process_batch(batch):
        # build tinker ModelInputs
        model_inputs = []
        for p in batch:
            messages = [{"role": "user", "content": p["prompt_text"]}]
            model_inputs.append(renderer.build_generation_prompt(messages))
            
        # sample
        tasks = []
        for minput in model_inputs:
            tasks.append(
                sample_with_retry(sampling_client, minput, num_samples, stop_condition)
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for p, result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"Error on {p['prompt_id']}: {result}")
                continue
                
            generations = []
            for seq in result.sequences:
                parsed_message, _ = renderer.parse_response(seq.tokens)
                generations.append(parsed_message["content"])
                
            out_obj = {
                "prompt_id": p["prompt_id"],
                "patient_id": p["patient_id"],
                "split": p["split"],
                "rarity_group": p["rarity_group"],
                "generations": generations
            }
            out_f.write(json.dumps(out_obj) + "\n")
            out_f.flush()

    for i in range(0, len(remaining_prompts), batch_size):
        batch = remaining_prompts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(remaining_prompts)-1)//batch_size + 1}...")
        await process_batch(batch)
        
    out_f.close()

async def main():
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        try:
            from config import TINKER_API_KEY
            api_key = TINKER_API_KEY
        except ImportError:
            pass
            
    if not api_key:
        print("ERROR: TINKER_API_KEY not found in environment variables or config.py!")
        print("Please export TINKER_API_KEY='your_key' or add it to src/utils/config.py.")
        sys.exit(1)
        
    os.environ["TINKER_API_KEY"] = api_key
    
    # Model identifiers
    M0 = "meta-llama/Llama-3.1-8B-Instruct"
    M1 = "tinker://9394d193-94dc-59ec-9b51-4eca06ebbc0f:train:0/sampler_weights/final"
    M2 = "tinker://90476feb-603f-550d-ab72-3560f27ee267:train:0/sampler_weights/final"
    
    # Load prompts
    prompts_path = Path("data/processed/eval_prompts.jsonl")
    if not prompts_path.exists():
        print(f"Could not find prompts at {prompts_path}")
        return
        
    prompts = []
    with open(prompts_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
            
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Running M0 (Base Model) ---")
    await generate_for_model(M0, True, prompts, out_dir / "M0_predictions.jsonl")
    
    print("--- Running M1 (Full SFT) ---")
    await generate_for_model(M1, False, prompts, out_dir / "M1_predictions.jsonl")
    
    print("--- Running M2 (Coarsened SFT) ---")
    await generate_for_model(M2, False, prompts, out_dir / "M2_predictions.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
