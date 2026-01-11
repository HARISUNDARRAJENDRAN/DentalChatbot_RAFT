"""
Model Loader for Training and Inference
"""

import torch
from pathlib import Path
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


def load_model_for_training(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantize: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    device_map: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model configured for QLoRA training.

    Args:
        model_name: HuggingFace model name or path
        quantize: Use 4-bit quantization
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")

    # Quantization config
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare for k-bit training
    if quantize:
        model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_model_for_inference(
    model_path: str,
    base_model_name: Optional[str] = None,
    quantize: bool = True,
    device_map: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load fine-tuned model for inference.

    Args:
        model_path: Path to fine-tuned model or HuggingFace repo
        base_model_name: Base model name (if loading LoRA adapter separately)
        quantize: Use 4-bit quantization
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_path}")

    # Check if this is a merged model or LoRA adapter
    model_path_obj = Path(model_path)
    is_lora = model_path_obj.exists() and (model_path_obj / "adapter_config.json").exists()

    # Quantization config
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        bnb_config = None

    if is_lora and base_model_name:
        # Load base model + LoRA adapter
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )

        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load merged model directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not quantize else None
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")
    return model, tokenizer


def merge_and_save(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: str
) -> None:
    """
    Merge LoRA weights with base model and save.

    Args:
        model: PEFT model with LoRA adapters
        tokenizer: Tokenizer
        output_path: Path to save merged model
    """
    print("Merging LoRA weights with base model...")

    # Merge weights
    merged_model = model.merge_and_unload()

    # Save
    print(f"Saving to: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Model saved!")


def push_to_hub(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    repo_name: str,
    private: bool = True,
    merge_first: bool = True
) -> str:
    """
    Push model to HuggingFace Hub.

    Args:
        model: Model to push
        tokenizer: Tokenizer
        repo_name: Repository name (e.g., "username/model-name")
        private: Make repository private
        merge_first: Merge LoRA weights before pushing

    Returns:
        Repository URL
    """
    if merge_first and hasattr(model, 'merge_and_unload'):
        print("Merging LoRA weights...")
        model = model.merge_and_unload()

    print(f"Pushing to: {repo_name}")
    model.push_to_hub(repo_name, private=private)
    tokenizer.push_to_hub(repo_name, private=private)

    url = f"https://huggingface.co/{repo_name}"
    print(f"Model pushed to: {url}")

    return url
