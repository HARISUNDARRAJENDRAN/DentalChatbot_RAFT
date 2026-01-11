"""
FAISS Retriever for Dental RAG
Handles document retrieval using FAISS vector similarity search.
"""

import pickle
from pathlib import Path
from dataclasses import dataclass
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e}\n"
        "Run: pip install faiss-cpu sentence-transformers"
    )


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with relevance score."""
    content: str
    source: str
    page_number: int
    chunk_id: str
    relevance_score: float


class FAISSRetriever:
    """
    FAISS-based document retriever for dental RAG.

    Uses cosine similarity with normalized embeddings for efficient retrieval.
    """

    def __init__(
        self,
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        device: str = "cpu"
    ):
        """
        Initialize the FAISS retriever.

        Args:
            embedding_model: HuggingFace model for embeddings
            device: Device for embedding model ("cpu" or "cuda")
        """
        self.device = device
        self.embedding_model_name = embedding_model

        print(f"Loading embedding model: {embedding_model}")
        self.embed_model = SentenceTransformer(embedding_model, device=device)
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()

        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict] = []

    def load_index(self, index_dir: Path | str) -> None:
        """
        Load pre-built FAISS index and chunk metadata.

        Args:
            index_dir: Directory containing index.faiss and chunks.pkl
        """
        index_dir = Path(index_dir)

        index_path = index_dir / "index.faiss"
        chunks_path = index_dir / "chunks.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks metadata not found: {chunks_path}")

        self.index = faiss.read_index(str(index_path))

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.chunks)} chunk metadata entries")

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build FAISS index from chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        self.chunks = chunks

        print(f"Computing embeddings for {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]

        embeddings = self.embed_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        print(f"FAISS index built with {self.index.ntotal} vectors")

    def save_index(self, output_dir: Path | str) -> None:
        """
        Save FAISS index and chunk metadata.

        Args:
            output_dir: Directory to save index files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(output_dir / "index.faiss"))

        with open(output_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Saved FAISS index to {output_dir}")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            k: Number of documents to retrieve
            score_threshold: Minimum relevance score (0-1)

        Returns:
            List of RetrievedDocument objects sorted by relevance
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        # Embed query
        query_embedding = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # Search FAISS
        scores, indices = self.index.search(query_embedding, k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            if score < score_threshold:
                continue

            chunk = self.chunks[idx]

            results.append(RetrievedDocument(
                content=chunk["text"],
                source=chunk["source"],
                page_number=chunk["page_number"],
                chunk_id=chunk["chunk_id"],
                relevance_score=float(score)
            ))

        return results

    def retrieve_batch(
        self,
        queries: list[str],
        k: int = 5
    ) -> list[list[RetrievedDocument]]:
        """
        Retrieve documents for multiple queries in batch.

        Args:
            queries: List of query texts
            k: Number of documents per query

        Returns:
            List of retrieval results per query
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        # Embed all queries
        query_embeddings = self.embed_model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # Batch search
        scores, indices = self.index.search(query_embeddings, k)

        # Build results
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx == -1:
                    continue

                chunk = self.chunks[idx]
                results.append(RetrievedDocument(
                    content=chunk["text"],
                    source=chunk["source"],
                    page_number=chunk["page_number"],
                    chunk_id=chunk["chunk_id"],
                    relevance_score=float(score)
                ))

            all_results.append(results)

        return all_results

    def get_stats(self) -> dict:
        """Get index statistics."""
        if self.index is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "num_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(self.chunks),
            "embedding_model": self.embedding_model_name
        }


def main():
    """Example usage of FAISS retriever."""
    retriever = FAISSRetriever(
        embedding_model="pritamdeka/S-PubMedBert-MS-MARCO"
    )

    index_dir = Path("data/embeddings/faiss_index")

    if index_dir.exists():
        retriever.load_index(index_dir)

        # Test retrieval
        query = "What are the indications for root canal therapy?"
        results = retriever.retrieve(query, k=5)

        print(f"\nQuery: {query}\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. [{doc.relevance_score:.3f}] {doc.source} (p.{doc.page_number})")
            print(f"   {doc.content[:150]}...\n")
    else:
        print(f"Index not found at {index_dir}")
        print("Run RAFT formatter first to create the index.")


if __name__ == "__main__":
    main()
