"""
Q&A Generator for Dental Content
Generates educational question-answer pairs from dental text chunks using LLMs.
"""

import json
import os
import time
from pathlib import Path
from typing import Generator
from dataclasses import dataclass, asdict
from tqdm import tqdm

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class QAPair:
    """Represents a question-answer pair with metadata."""
    question: str
    answer: str
    difficulty: str
    source_chunk_id: str
    source: str
    page_number: int
    reasoning: str = ""


# Prompt template for Q&A generation
QA_GENERATION_PROMPT = """You are a dental education expert. Generate educational questions from this dental textbook excerpt that would help dental students prepare for exams.

TEXTBOOK EXCERPT:
{chunk_text}

SOURCE: {source}, Page {page_number}

Generate 3 questions with detailed answers. For each question:
1. Make it specific and educational (not too broad)
2. Include chain-of-thought reasoning in the answer
3. Reference specific facts from the text
4. Indicate difficulty level (basic/intermediate/advanced)

Respond in JSON format:
{{
    "questions": [
        {{
            "question": "What is...",
            "answer": "Based on the text, ... First, we can see that... Therefore...",
            "difficulty": "intermediate",
            "reasoning": "This tests understanding of..."
        }}
    ]
}}

Generate exactly 3 questions. Ensure answers are comprehensive and cite specific information from the text."""


class QAGenerator:
    """
    Generate Q&A pairs from dental text chunks using LLMs.

    Supports both Groq (free, Llama 3.1 70B) and OpenAI (GPT-4).
    """

    def __init__(
        self,
        provider: str = "groq",
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Q&A generator.

        Args:
            provider: LLM provider - "groq" (free) or "openai"
            model: Model name (default: llama-3.1-70b-versatile for groq, gpt-4-turbo for openai)
            api_key: API key (or set GROQ_API_KEY / OPENAI_API_KEY env var)
            max_retries: Maximum retry attempts on API errors
            retry_delay: Delay between retries in seconds
        """
        self.provider = provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if provider == "groq":
            if Groq is None:
                raise ImportError("groq package not installed. Run: pip install groq")

            api_key = api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")

            self.client = Groq(api_key=api_key)
            self.model = model or "llama-3.1-70b-versatile"

        elif provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package not installed. Run: pip install openai")

            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.client = OpenAI(api_key=api_key)
            self.model = model or "gpt-4-turbo"

        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'groq' or 'openai'")

    def generate_qa_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        source: str,
        page_number: int
    ) -> list[QAPair]:
        """
        Generate Q&A pairs from a single text chunk.

        Args:
            chunk_text: The text content of the chunk
            chunk_id: Unique identifier for the chunk
            source: Source document name
            page_number: Page number in source

        Returns:
            List of QAPair objects
        """
        prompt = QA_GENERATION_PROMPT.format(
            chunk_text=chunk_text,
            source=source,
            page_number=page_number
        )

        for attempt in range(self.max_retries):
            try:
                if self.provider == "groq":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=2048,
                        response_format={"type": "json_object"}
                    )
                else:  # openai
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=2048,
                        response_format={"type": "json_object"}
                    )

                content = response.choices[0].message.content
                data = json.loads(content)

                qa_pairs = []
                for q in data.get("questions", []):
                    qa_pair = QAPair(
                        question=q["question"],
                        answer=q["answer"],
                        difficulty=q.get("difficulty", "intermediate"),
                        source_chunk_id=chunk_id,
                        source=source,
                        page_number=page_number,
                        reasoning=q.get("reasoning", "")
                    )
                    qa_pairs.append(qa_pair)

                return qa_pairs

            except json.JSONDecodeError as e:
                print(f"JSON parse error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue

            except Exception as e:
                print(f"API error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                continue

        return []  # Return empty list if all retries failed

    def process_chunks_file(
        self,
        input_path: Path | str,
        output_path: Path | str,
        max_chunks: int | None = None,
        show_progress: bool = True
    ) -> int:
        """
        Process chunks JSONL file and generate Q&A pairs.

        Args:
            input_path: Path to chunks JSONL file
            output_path: Path to save Q&A pairs JSONL
            max_chunks: Maximum chunks to process (for testing)
            show_progress: Show progress bar

        Returns:
            Total number of Q&A pairs generated
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Count lines for progress
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        if max_chunks:
            total_lines = min(total_lines, max_chunks)

        total_qa = 0
        processed = 0

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            iterator = tqdm(fin, total=total_lines, desc="Generating Q&A") if show_progress else fin

            for line in iterator:
                if max_chunks and processed >= max_chunks:
                    break

                chunk = json.loads(line)

                qa_pairs = self.generate_qa_from_chunk(
                    chunk_text=chunk["text"],
                    chunk_id=chunk["chunk_id"],
                    source=chunk["source"],
                    page_number=chunk["page_number"]
                )

                for qa in qa_pairs:
                    fout.write(json.dumps(asdict(qa), ensure_ascii=False) + "\n")
                    total_qa += 1

                processed += 1

                # Rate limiting for API calls
                time.sleep(0.1)

        print(f"Generated {total_qa} Q&A pairs from {processed} chunks")
        return total_qa

    def iter_qa_pairs(
        self,
        input_path: Path | str
    ) -> Generator[QAPair, None, None]:
        """
        Iterate over Q&A pairs from a JSONL file.

        Args:
            input_path: Path to Q&A pairs JSONL file

        Yields:
            QAPair objects
        """
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield QAPair(**data)


def main():
    """Example usage of Q&A generator."""
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("Set GROQ_API_KEY environment variable first.")
        print("Get free API key at: https://console.groq.com/keys")
        return

    generator = QAGenerator(provider="groq")

    input_path = Path("data/processed/chunks/chunks.jsonl")
    output_path = Path("data/processed/qa_pairs/qa_pairs.jsonl")

    if input_path.exists():
        # Process first 10 chunks for testing
        total = generator.process_chunks_file(
            input_path=input_path,
            output_path=output_path,
            max_chunks=10,  # Remove this limit for full processing
            show_progress=True
        )
        print(f"Generated {total} Q&A pairs")
    else:
        print(f"Input file {input_path} not found. Run chunking first.")


if __name__ == "__main__":
    main()
