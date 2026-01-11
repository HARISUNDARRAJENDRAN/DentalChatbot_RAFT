"""
RAFT Dataset Formatter
Formats Q&A pairs with oracle and distractor documents for RAFT training.
"""

import json
import random
import pickle
from pathlib import Path
from typing import Generator
from dataclasses import dataclass, asdict, field
import numpy as np
from tqdm import tqdm

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install faiss-cpu sentence-transformers")


@dataclass
class ContextDocument:
    """Represents a context document (oracle or distractor)."""
    content: str
    source: str
    page_number: int
    is_oracle: bool


@dataclass
class RAFTExample:
    """Represents a single RAFT training example."""
    question: str
    context: list[ContextDocument]
    answer: str
    metadata: dict = field(default_factory=dict)


class RAFTFormatter:
    """
    Format Q&A pairs into RAFT training examples.

    RAFT methodology:
    - P% of examples include oracle document + distractors
    - (1-P)% include only distractors (teach model to handle missing info)
    - Uses ##begin_quote## and ##end_quote## markers for citations
    """

    def __init__(
        self,
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        oracle_ratio: float = 0.8,
        num_distractors: int = 4,
        device: str = "cpu"
    ):
        """
        Initialize RAFT formatter.

        Args:
            embedding_model: HuggingFace model for embeddings
            oracle_ratio: Probability of including oracle document (P)
            num_distractors: Number of distractor documents per example
            device: Device for embedding model ("cpu" or "cuda")
        """
        self.oracle_ratio = oracle_ratio
        self.num_distractors = num_distractors

        print(f"Loading embedding model: {embedding_model}")
        self.embed_model = SentenceTransformer(embedding_model, device=device)
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()

        self.chunks: list[dict] = []
        self.chunk_embeddings: np.ndarray | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None

    def load_chunks(self, chunks_path: Path | str) -> None:
        """
        Load chunks from JSONL and build FAISS index.

        Args:
            chunks_path: Path to chunks JSONL file
        """
        chunks_path = Path(chunks_path)
        print(f"Loading chunks from {chunks_path}")

        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        print(f"Loaded {len(self.chunks)} chunks")
        self._build_index()

    def _build_index(self) -> None:
        """Build FAISS index from chunks."""
        print("Computing embeddings for all chunks...")

        texts = [c["text"] for c in self.chunks]
        self.chunk_embeddings = self.embed_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )

        # Build FAISS index (inner product = cosine similarity for normalized vectors)
        print("Building FAISS index...")
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(self.chunk_embeddings.astype(np.float32))
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    def save_index(self, output_dir: Path | str) -> None:
        """
        Save FAISS index and chunk metadata.

        Args:
            output_dir: Directory to save index files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.faiss_index, str(output_dir / "index.faiss"))

        with open(output_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Saved FAISS index to {output_dir}")

    def load_index(self, index_dir: Path | str) -> None:
        """
        Load pre-built FAISS index.

        Args:
            index_dir: Directory containing index files
        """
        index_dir = Path(index_dir)

        self.faiss_index = faiss.read_index(str(index_dir / "index.faiss"))

        with open(index_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")

    def find_distractors(
        self,
        question: str,
        oracle_chunk_id: str,
        k: int = 20
    ) -> list[dict]:
        """
        Find distractor chunks for a question.

        Args:
            question: The question text
            oracle_chunk_id: ID of oracle chunk to exclude
            k: Number of candidates to retrieve

        Returns:
            List of distractor chunk dictionaries
        """
        # Embed question
        query_embedding = self.embed_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # Search FAISS
        distances, indices = self.faiss_index.search(query_embedding, k)

        # Select distractors (exclude oracle, mix of similar and random)
        distractors = []
        for idx in indices[0]:
            if len(distractors) >= self.num_distractors:
                break

            chunk = self.chunks[idx]
            if chunk["chunk_id"] != oracle_chunk_id:
                distractors.append(chunk)

        # If not enough similar distractors, add random ones
        while len(distractors) < self.num_distractors:
            random_idx = random.randint(0, len(self.chunks) - 1)
            chunk = self.chunks[random_idx]
            if chunk["chunk_id"] != oracle_chunk_id and chunk not in distractors:
                distractors.append(chunk)

        return distractors[:self.num_distractors]

    def format_answer_with_citations(
        self,
        answer: str,
        oracle_text: str | None
    ) -> str:
        """
        Format answer with citation markers.

        Args:
            answer: Original answer text
            oracle_text: Oracle document text (for citation extraction)

        Returns:
            Answer with ##begin_quote## and ##end_quote## markers
        """
        if oracle_text is None:
            # No oracle - add disclaimer
            return f"Based on the provided documents, I cannot find sufficient information to fully answer this question.\n\n{answer}"

        # Find overlapping phrases between answer and oracle
        # For simplicity, we'll add citations around key phrases
        # In production, you'd want more sophisticated citation extraction

        # Extract a relevant quote from oracle (first 100 chars of meaningful content)
        oracle_preview = oracle_text[:200].strip()
        if len(oracle_text) > 200:
            oracle_preview += "..."

        formatted = f"Based on the provided context, ##begin_quote##{oracle_preview}##end_quote##\n\n{answer}"

        return formatted

    def create_raft_example(
        self,
        qa_pair: dict,
        oracle_chunk: dict
    ) -> RAFTExample:
        """
        Create a single RAFT training example.

        Args:
            qa_pair: Q&A pair dictionary
            oracle_chunk: The oracle chunk for this Q&A

        Returns:
            RAFTExample object
        """
        include_oracle = random.random() < self.oracle_ratio

        # Get distractors
        distractors = self.find_distractors(
            question=qa_pair["question"],
            oracle_chunk_id=oracle_chunk["chunk_id"]
        )

        # Build context
        context = []

        if include_oracle:
            context.append(ContextDocument(
                content=oracle_chunk["text"],
                source=oracle_chunk["source"],
                page_number=oracle_chunk["page_number"],
                is_oracle=True
            ))

        for dist in distractors:
            context.append(ContextDocument(
                content=dist["text"],
                source=dist["source"],
                page_number=dist["page_number"],
                is_oracle=False
            ))

        # Shuffle context so oracle isn't always first
        random.shuffle(context)

        # Format answer with citations
        answer = self.format_answer_with_citations(
            answer=qa_pair["answer"],
            oracle_text=oracle_chunk["text"] if include_oracle else None
        )

        return RAFTExample(
            question=qa_pair["question"],
            context=context,
            answer=answer,
            metadata={
                "difficulty": qa_pair.get("difficulty", "intermediate"),
                "source_chunk_id": qa_pair["source_chunk_id"],
                "includes_oracle": include_oracle
            }
        )

    def process_qa_file(
        self,
        qa_path: Path | str,
        output_path: Path | str,
        chunks_path: Path | str | None = None,
        train_split: float = 0.9,
        show_progress: bool = True
    ) -> tuple[int, int]:
        """
        Process Q&A pairs into RAFT training dataset.

        Args:
            qa_path: Path to Q&A pairs JSONL
            output_path: Base path for output (will create train.jsonl and val.jsonl)
            chunks_path: Path to chunks JSONL (if not already loaded)
            train_split: Fraction of data for training
            show_progress: Show progress bar

        Returns:
            Tuple of (train_count, val_count)
        """
        qa_path = Path(qa_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load chunks if not already loaded
        if chunks_path and not self.chunks:
            self.load_chunks(chunks_path)

        # Build chunk lookup by ID
        chunk_lookup = {c["chunk_id"]: c for c in self.chunks}

        # Load Q&A pairs
        qa_pairs = []
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                qa_pairs.append(json.loads(line))

        print(f"Processing {len(qa_pairs)} Q&A pairs...")

        # Shuffle and split
        random.shuffle(qa_pairs)
        split_idx = int(len(qa_pairs) * train_split)

        train_pairs = qa_pairs[:split_idx]
        val_pairs = qa_pairs[split_idx:]

        # Process train set
        train_path = output_path / "train.jsonl"
        train_count = self._process_split(
            qa_pairs=train_pairs,
            output_path=train_path,
            chunk_lookup=chunk_lookup,
            desc="Processing train",
            show_progress=show_progress
        )

        # Process val set
        val_path = output_path / "val.jsonl"
        val_count = self._process_split(
            qa_pairs=val_pairs,
            output_path=val_path,
            chunk_lookup=chunk_lookup,
            desc="Processing val",
            show_progress=show_progress
        )

        print(f"Created {train_count} training examples, {val_count} validation examples")
        return train_count, val_count

    def _process_split(
        self,
        qa_pairs: list[dict],
        output_path: Path,
        chunk_lookup: dict,
        desc: str,
        show_progress: bool
    ) -> int:
        """Process a split of Q&A pairs."""
        count = 0

        with open(output_path, "w", encoding="utf-8") as f:
            iterator = tqdm(qa_pairs, desc=desc) if show_progress else qa_pairs

            for qa in iterator:
                chunk_id = qa["source_chunk_id"]

                if chunk_id not in chunk_lookup:
                    continue

                oracle_chunk = chunk_lookup[chunk_id]

                example = self.create_raft_example(qa, oracle_chunk)

                # Convert to serializable format
                example_dict = {
                    "question": example.question,
                    "context": [asdict(c) for c in example.context],
                    "answer": example.answer,
                    "metadata": example.metadata
                }

                f.write(json.dumps(example_dict, ensure_ascii=False) + "\n")
                count += 1

        return count

    def iter_examples(
        self,
        input_path: Path | str
    ) -> Generator[RAFTExample, None, None]:
        """
        Iterate over RAFT examples from a JSONL file.

        Args:
            input_path: Path to RAFT examples JSONL

        Yields:
            RAFTExample objects
        """
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                context = [
                    ContextDocument(**c)
                    for c in data["context"]
                ]
                yield RAFTExample(
                    question=data["question"],
                    context=context,
                    answer=data["answer"],
                    metadata=data.get("metadata", {})
                )


def main():
    """Example usage of RAFT formatter."""
    formatter = RAFTFormatter(
        embedding_model="pritamdeka/S-PubMedBert-MS-MARCO",
        oracle_ratio=0.8,
        num_distractors=4
    )

    chunks_path = Path("data/processed/chunks/chunks.jsonl")
    qa_path = Path("data/processed/qa_pairs/qa_pairs.jsonl")
    output_path = Path("data/processed/raft_dataset")

    if chunks_path.exists() and qa_path.exists():
        # Load chunks and build index
        formatter.load_chunks(chunks_path)

        # Save index for later use
        formatter.save_index(Path("data/embeddings/faiss_index"))

        # Process Q&A into RAFT format
        train_count, val_count = formatter.process_qa_file(
            qa_path=qa_path,
            output_path=output_path,
            train_split=0.9,
            show_progress=True
        )

        print(f"Done! Train: {train_count}, Val: {val_count}")
    else:
        print("Input files not found. Run chunking and Q&A generation first.")


if __name__ == "__main__":
    main()
