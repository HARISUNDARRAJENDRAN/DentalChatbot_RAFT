"""
Training Configuration for RAFT Fine-Tuning
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    quantization: str = "4bit"  # "4bit", "8bit", or "none"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration for QLoRA training."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Output
    output_dir: str = "./checkpoints"
    run_name: str = "dental-raft"

    # Training schedule
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # Learning rate
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100

    # Optimization
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500

    # Memory
    gradient_checkpointing: bool = True
    max_seq_length: int = 2048

    # Misc
    seed: int = 42
    report_to: str = "wandb"  # or "none"

    # Dataset
    packing: bool = False


@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str = "data/processed/raft_dataset/train.jsonl"
    val_path: str = "data/processed/raft_dataset/val.jsonl"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 100

    # RAFT
    oracle_ratio: float = 0.8
    num_distractors: int = 4

    # Embedding
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def load_model_for_training(
    model_config: ModelConfig,
    lora_config: LoRAConfig
):
    """
    Load model configured for QLoRA training.

    Args:
        model_config: Model configuration
        lora_config: LoRA configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Quantization config
    if model_config.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=get_torch_dtype(model_config.torch_dtype)
        )
    elif model_config.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        bnb_config = None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=get_torch_dtype(model_config.torch_dtype) if bnb_config is None else None
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare for training
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type
    )

    model = get_peft_model(model, peft_config)

    return model, tokenizer
