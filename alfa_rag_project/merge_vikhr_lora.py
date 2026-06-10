"""
Merge Vikhr LoRA adapter into base model.

Usage:
    python merge_vikhr_lora.py --save-dir data/merged_models/vikhrllama1B_AlfaBank_merged --push-repo lirex111/vikhrllama1B_AlfaBank_merged

What it does:
1. Loads base Vikhr model: Vikhrmodels/Vikhr-Llama-3.2-1B-instruct
2. Loads LoRA adapter: lirex111/vikhrllama1B_AlfaBank
3. Merges adapter into base model
4. Saves flat model locally
5. Optionally pushes to Hugging Face Hub
"""

import argparse
import gc
import os
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from huggingface_hub import HfApi, login


BASE_MODEL_ID = "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct"
ADAPTER_ID = "lirex111/vikhrllama1B_AlfaBank"


def merge_and_save(
    save_dir: str,
    push_repo: str | None = None,
    hf_token: str | None = None,
    low_cpu_mem_usage: bool = True,
) -> None:
    """
    Merge LoRA adapter into base model, save locally, and optionally push to HF Hub.

    Args:
        save_dir: Local path to save merged model.
        push_repo: Optional Hugging Face repo ID, e.g. "lirex111/vikhrllama1B_AlfaBank_merged".
        hf_token: Optional HF token for push. If None, uses env HUGGINGFACE_TOKEN or cache.
        low_cpu_mem_usage: Use low_cpu_mem_usage for model loading (saves RAM).
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if hf_token:
        login(token=hf_token)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading base model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=low_cpu_mem_usage,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[2/5] Loading adapter: {ADAPTER_ID}")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_ID,
        torch_dtype=torch.float16,
        is_trainable=False,
    )

    print("[3/5] Merging adapter into base model")
    merged_model = model.merge_and_unload()
    merged_model.eval()

    # Free memory before saving
    del base_model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[4/5] Saving tokenizer and merged model to {save_path}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    merged_model.save_pretrained(save_path, safe_serialization=True)

    # Free memory after saving
    del merged_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[4/5] Local save complete: {save_path}")

    if push_repo:
        print(f"[5/5] Pushing merged model to Hugging Face: {push_repo}")
        api = HfApi()
        # Создаём репо, если его ещё нет
        try:
            api.create_repo(repo_id=push_repo, repo_type="model", exist_ok=True)
            print(f"[5/5] Repo ready: https://huggingface.co/{push_repo}")
        except Exception as e:
            print(f"[5/5] Repo creation warning: {e}")
        api.upload_folder(
            folder_path=str(save_path),
            repo_id=push_repo,
            repo_type="model",
        )
        print(f"[5/5] Push complete: https://huggingface.co/{push_repo}")
    else:
        print("[5/5] Push skipped (no --push-repo provided)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Vikhr LoRA adapter into base model")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/merged_models/vikhrllama1B_AlfaBank_merged",
        help="Local directory to save merged model",
    )
    parser.add_argument(
        "--push-repo",
        type=str,
        default=None,
        help="Hugging Face repo ID to push merged model, e.g. 'lirex111/vikhrllama1B_AlfaBank_merged'",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token. If None, uses HUGGINGFACE_TOKEN env var or cache.",
    )
    parser.add_argument(
        "--full-cpu-mem",
        action="store_true",
        help="Disable low_cpu_mem_usage (use more RAM but may be faster)",
    )
    args = parser.parse_args()

    merge_and_save(
        save_dir=args.save_dir,
        push_repo=args.push_repo,
        hf_token=args.hf_token,
        low_cpu_mem_usage=not args.full_cpu_mem,
    )


if __name__ == "__main__":
    main()
