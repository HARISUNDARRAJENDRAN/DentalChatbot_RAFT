"""
LLM Generator for Dental RAG
Handles text generation using fine-tuned Llama model.
"""

from pathlib import Path
from dataclasses import dataclass
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e}\n"
        "Run: pip install transformers accelerate bitsandbytes"
    )


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


# System prompt for dental assistant
DENTAL_SYSTEM_PROMPT = """You are a dental education assistant helping dental students learn.
Answer questions using the provided documents.
- Cite sources using ##begin_quote## and ##end_quote## markers
- If the documents don't contain relevant information, say so clearly
- Provide detailed, educational explanations
- Focus on accuracy and cite specific facts from the documents"""


class DentalGenerator:
    """
    LLM-based text generator for dental chatbot.

    Supports loading fine-tuned models with optional quantization.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        quantize: bool = True,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the generator.

        Args:
            model_name: HuggingFace model name or local path
            quantize: Use 4-bit quantization for lower memory
            device_map: Device mapping strategy
            torch_dtype: Tensor dtype for model
        """
        self.model_name = model_name
        self.device_map = device_map

        print(f"Loading model: {model_name}")

        # Setup quantization if requested
        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded. Device: {next(self.model.parameters()).device}")

    def format_prompt(
        self,
        question: str,
        context_docs: list[dict],
        system_prompt: str = DENTAL_SYSTEM_PROMPT
    ) -> str:
        """
        Format prompt for Llama 3.1 Instruct format.

        Args:
            question: User question
            context_docs: List of context documents with 'content', 'source', 'page_number'
            system_prompt: System prompt

        Returns:
            Formatted prompt string
        """
        # Format context documents
        context_str = "\n\n".join([
            f"Document {i+1} ({doc['source']}, p.{doc['page_number']}):\n{doc['content']}"
            for i, doc in enumerate(context_docs)
        ])

        # Llama 3.1 Instruct format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Question: {question}

Documents:
{context_str}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

"""
        return prompt

    def generate(
        self,
        question: str,
        context_docs: list[dict],
        config: GenerationConfig | None = None
    ) -> str:
        """
        Generate answer for a question given context documents.

        Args:
            question: User question
            context_docs: List of retrieved context documents
            config: Generation configuration

        Returns:
            Generated answer text
        """
        if config is None:
            config = GenerationConfig()

        # Format prompt
        prompt = self.format_prompt(question, context_docs)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            answer = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        else:
            # Fallback: get text after the prompt
            answer = full_response[len(prompt):] if len(full_response) > len(prompt) else full_response

        return answer.strip()

    def generate_batch(
        self,
        questions: list[str],
        context_docs_list: list[list[dict]],
        config: GenerationConfig | None = None
    ) -> list[str]:
        """
        Generate answers for multiple questions.

        Args:
            questions: List of questions
            context_docs_list: List of context document lists
            config: Generation configuration

        Returns:
            List of generated answers
        """
        answers = []
        for question, context_docs in zip(questions, context_docs_list):
            answer = self.generate(question, context_docs, config)
            answers.append(answer)

        return answers

    def clear_cache(self) -> None:
        """Clear CUDA cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Example usage of dental generator."""
    # This requires a GPU with sufficient memory
    print("Initializing generator...")

    try:
        generator = DentalGenerator(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            quantize=True
        )

        # Test with sample data
        question = "What are the indications for root canal therapy?"
        context_docs = [
            {
                "content": "Root canal therapy is indicated when the dental pulp becomes irreversibly inflamed or necrotic. Common causes include deep caries, trauma, or repeated dental procedures.",
                "source": "Endodontics_Textbook.pdf",
                "page_number": 45
            },
            {
                "content": "Signs that indicate the need for root canal treatment include persistent tooth pain, prolonged sensitivity to hot or cold, and discoloration of the tooth.",
                "source": "Clinical_Dentistry.pdf",
                "page_number": 112
            }
        ]

        print(f"\nQuestion: {question}\n")
        answer = generator.generate(question, context_docs)
        print(f"Answer:\n{answer}")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires a GPU with at least 8GB VRAM for quantized model.")


if __name__ == "__main__":
    main()
