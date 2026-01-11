"""
RAFT Trainer
Custom trainer for RAFT fine-tuning with dental data.
"""

import json
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

import torch
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from datasets import Dataset


@dataclass
class RAFTTrainerConfig:
    """Configuration for RAFT trainer."""
    output_dir: str = "./checkpoints"
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "wandb"


def load_raft_dataset(file_path: Path | str) -> Dataset:
    """
    Load RAFT dataset from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        HuggingFace Dataset
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return Dataset.from_list(data)


def create_formatting_function(tokenizer) -> Callable:
    """
    Create formatting function for RAFT examples.

    Args:
        tokenizer: Tokenizer for the model

    Returns:
        Formatting function
    """
    def format_raft_prompt(example):
        """Format RAFT example into Llama 3.1 Instruct format."""
        # Format context documents
        context_parts = []
        for i, doc in enumerate(example['context']):
            context_parts.append(
                f"Document {i+1} ({doc['source']}, p.{doc['page_number']}):\n{doc['content']}"
            )
        context_str = "\n\n".join(context_parts)

        # Llama 3.1 Instruct format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a dental education assistant. Answer questions using the provided documents. Cite sources using ##begin_quote## and ##end_quote## markers. If documents don't contain relevant information, say so clearly.<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Question: {example['question']}

Documents:
{context_str}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

{example['answer']}<|eot_id|>"""

        return prompt

    return format_raft_prompt


class RAFTTrainer:
    """
    Trainer for RAFT fine-tuning.

    Wraps HuggingFace SFTTrainer with RAFT-specific configuration.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[RAFTTrainerConfig] = None
    ):
        """
        Initialize RAFT trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or RAFTTrainerConfig()

        self._setup_trainer()

    def _setup_trainer(self) -> None:
        """Setup the SFTTrainer."""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,

            # Training schedule
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,

            # Learning rate
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=self.config.warmup_steps,

            # Optimization
            optim="paged_adamw_8bit",
            fp16=False,
            bf16=self.config.use_bf16,

            # Logging and saving
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,

            # Evaluation
            evaluation_strategy="steps" if self.eval_dataset else "no",
            eval_steps=self.config.eval_steps if self.eval_dataset else None,

            # Memory
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_grad_norm=0.3,

            # Misc
            report_to=self.config.report_to,
            seed=42,
        )

        # Create formatting function
        formatting_func = create_formatting_function(self.tokenizer)

        # Initialize SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            formatting_func=formatting_func,
            max_seq_length=self.config.max_seq_length,
            packing=False,
        )

    def train(self) -> None:
        """Run training."""
        print("Starting training...")
        self.trainer.train()

    def save(self, output_path: str) -> None:
        """Save the trained model."""
        print(f"Saving model to: {output_path}")
        self.trainer.save_model(output_path)

    def evaluate(self) -> dict:
        """Run evaluation."""
        if self.eval_dataset is None:
            raise ValueError("No evaluation dataset provided")

        return self.trainer.evaluate()


def main():
    """Example usage of RAFT trainer."""
    from .model_loader import load_model_for_training

    # Paths
    train_path = Path("data/processed/raft_dataset/train.jsonl")
    val_path = Path("data/processed/raft_dataset/val.jsonl")

    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        return

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_raft_dataset(train_path)
    val_dataset = load_raft_dataset(val_path) if val_path.exists() else None

    print(f"Train examples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val examples: {len(val_dataset)}")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_for_training(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        quantize=True,
        lora_r=16,
        lora_alpha=32
    )

    # Configure training
    config = RAFTTrainerConfig(
        output_dir="./checkpoints/dental-raft",
        num_epochs=3,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_seq_length=2048
    )

    # Initialize trainer
    trainer = RAFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config
    )

    # Train
    trainer.train()

    # Save
    trainer.save("./final_model")

    print("Training complete!")


if __name__ == "__main__":
    main()
