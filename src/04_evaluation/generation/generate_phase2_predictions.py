import os
import sys
import json
import asyncio
import tenacity
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

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

async def generate_for_model(model_name_or_id, is_base_model, prompts_file, output_file, num_samples=10, batch_size=50):
    if not prompts_file.exists():
        print(f"Could not find prompts at {prompts_file}")
        return

    prompts = []
    with open(prompts_file, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

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
    
    print(f"Generating for {model_name_or_id} on {prompts_file.name}. {len(completed_ids)} already done, {len(remaining_prompts)} remaining.")
    if not remaining_prompts:
        return
        
    out_f = open(output_file, "a")
    
    async def process_batch(batch):
        model_inputs = []
        for p in batch:
            messages = [{"role": "user", "content": p["prompt_text"]}]
            model_inputs.append(renderer.build_generation_prompt(messages))
            
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
                "target_text": p["target_text"], # Make sure we save the target!
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
    import json
    
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        try:
            from config import TINKER_API_KEY
            api_key = TINKER_API_KEY
        except ImportError:
            pass
            
    if not api_key:
        print("ERROR: TINKER_API_KEY not found in environment variables or config.py!")
        sys.exit(1)
        
    os.environ["TINKER_API_KEY"] = api_key
    
    # Extracted Model IDs from the successful Tinker runs
    # (Logs showed these URIs at checkpoints)
    M1_12_EPOCH = "tinker://3b61c546-ea00-56ea-a1f3-93f3576dfd34:train:0/sampler_weights/final" 
    M2_12_EPOCH = "tinker://35dbde86-8ca9-5616-987c-c8245d8f381b:train:0/sampler_weights/final"
    M0 = "meta-llama/Llama-3.1-8B-Instruct"
    
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    attacks = [
        ("gene", Path("data/processed/eval_prompts_gene_attack.jsonl")),
        ("size", Path("data/processed/eval_prompts_size_attack.jsonl"))
    ]
    
    models = [
        ("M0_baseline", M0, True),
        ("M1_exact_12epo", M1_12_EPOCH, False),
        ("M2_coars_12epo", M2_12_EPOCH, False)
    ]
    
    # We will generate 10 samples per prompt to test for exact string extraction
    NUM_SAMPLES = 10 
    
    for attack_name, prompt_file in attacks:
        print(f"\n======================================")
        print(f"Starting {attack_name.upper()} Attack Suite")
        print(f"======================================")
        
        for friendly_name, model_uri, is_base in models:
            out_file = out_dir / f"{friendly_name}_{attack_name}_predictions.jsonl"
            print(f"\n--- Running {friendly_name} ---")
            await generate_for_model(model_uri, is_base, prompt_file, out_file, num_samples=NUM_SAMPLES)

if __name__ == "__main__":
    asyncio.run(main())
