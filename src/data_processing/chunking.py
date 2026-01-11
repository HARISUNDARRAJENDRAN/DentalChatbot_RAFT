"""
Text Chunking for Dental Documents
Implements semantic chunking with overlap for optimal retrieval.
"""

import json
from pathlib import Path
from typing import Generator
from dataclasses import dataclass, asdict
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


@dataclass
class TextChunk:
    """Represents a single text chunk with metadata."""
    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    total_chunks_in_page: int
    token_count: int


class TextChunker:
    """
    Chunk dental document text for embedding and retrieval.

    Uses RecursiveCharacterTextSplitter with token-based chunking
    for optimal semantic coherence.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        model_name: str = "gpt-4"
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            model_name: Model name for tokenizer (used for token counting)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,  # Approximate chars (4 chars per token)
            chunk_overlap=chunk_overlap * 4,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str, source: str,
                   page_number: int) -> list[TextChunk]:
        """
        Split text into chunks with metadata.

        Args:
            text: Text to chunk
            source: Source document name
            page_number: Page number in source document

        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) < 50:
            return []

        # Split text into chunks
        chunks_text = self.splitter.split_text(text)

        # Create chunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            chunk = TextChunk(
                chunk_id=f"{source}_{page_number}_{idx}",
                text=chunk_text,
                source=source,
                page_number=page_number,
                chunk_index=idx,
                total_chunks_in_page=len(chunks_text),
                token_count=self._count_tokens(chunk_text)
            )
            chunks.append(chunk)

        return chunks

    def chunk_pages_file(
        self,
        input_path: Path | str,
        output_path: Path | str,
        show_progress: bool = True
    ) -> int:
        """
        Process extracted pages JSONL and create chunks.

        Args:
            input_path: Path to extracted pages JSONL
            output_path: Path to save chunks JSONL
            show_progress: Show progress bar

        Returns:
            Total number of chunks created
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Count lines for progress bar
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        total_chunks = 0

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            iterator = tqdm(fin, total=total_lines, desc="Chunking") if show_progress else fin

            for line in iterator:
                page = json.loads(line)

                chunks = self.chunk_text(
                    text=page["text"],
                    source=page["source"],
                    page_number=page["page_number"]
                )

                for chunk in chunks:
                    fout.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                    total_chunks += 1

        print(f"Created {total_chunks} chunks, saved to {output_path}")
        return total_chunks

    def iter_chunks(
        self,
        input_path: Path | str
    ) -> Generator[TextChunk, None, None]:
        """
        Iterate over chunks from a chunks JSONL file.

        Args:
            input_path: Path to chunks JSONL file

        Yields:
            TextChunk objects
        """
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield TextChunk(**data)


class SentenceWindowChunker:
    """
    Alternative chunking strategy: sentence-level with window expansion.

    Stores individual sentences but retrieves with surrounding context.
    Better for precise retrieval of specific facts.
    """

    def __init__(self, window_size: int = 3):
        """
        Initialize sentence window chunker.

        Args:
            window_size: Number of sentences to include on each side
        """
        self.window_size = window_size

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with spacy/nltk)
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in ".!?" and len(current.strip()) > 10:
                sentences.append(current.strip())
                current = ""

        if current.strip():
            sentences.append(current.strip())

        return sentences

    def get_window(self, sentences: list[str], index: int) -> str:
        """Get sentence with surrounding window context."""
        start = max(0, index - self.window_size)
        end = min(len(sentences), index + self.window_size + 1)
        return " ".join(sentences[start:end])


def main():
    """Example usage of text chunker."""
    chunker = TextChunker(chunk_size=512, chunk_overlap=100)

    input_path = Path("data/processed/chunks/raw_pages.jsonl")
    output_path = Path("data/processed/chunks/chunks.jsonl")

    if input_path.exists():
        total = chunker.chunk_pages_file(
            input_path=input_path,
            output_path=output_path,
            show_progress=True
        )
        print(f"Total chunks: {total}")
    else:
        print(f"Input file {input_path} not found. Run PDF extraction first.")


if __name__ == "__main__":
    main()
