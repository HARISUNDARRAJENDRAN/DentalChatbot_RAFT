"""
RAG Pipeline for Dental Chatbot
Orchestrates retrieval and generation for end-to-end Q&A.
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable

from .retriever import FAISSRetriever, RetrievedDocument
from .generator import DentalGenerator, GenerationConfig


@dataclass
class RAGResponse:
    """Complete RAG response with answer and sources."""
    question: str
    answer: str
    sources: list[RetrievedDocument]
    formatted_sources: str


class RAGPipeline:
    """
    End-to-end RAG pipeline for dental chatbot.

    Combines FAISS retrieval with LLM generation.
    """

    def __init__(
        self,
        retriever: FAISSRetriever | None = None,
        generator: DentalGenerator | None = None,
        num_docs: int = 5,
        score_threshold: float = 0.3
    ):
        """
        Initialize RAG pipeline.

        Args:
            retriever: Pre-initialized FAISS retriever
            generator: Pre-initialized LLM generator
            num_docs: Number of documents to retrieve
            score_threshold: Minimum relevance score for documents
        """
        self.retriever = retriever
        self.generator = generator
        self.num_docs = num_docs
        self.score_threshold = score_threshold

    @classmethod
    def from_config(
        cls,
        index_dir: Path | str,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        quantize: bool = True,
        device: str = "auto",
        num_docs: int = 5
    ) -> "RAGPipeline":
        """
        Create pipeline from configuration.

        Args:
            index_dir: Directory containing FAISS index
            model_name: LLM model name or path
            embedding_model: Embedding model for retrieval
            quantize: Use 4-bit quantization
            device: Device for models
            num_docs: Number of documents to retrieve

        Returns:
            Configured RAGPipeline instance
        """
        # Initialize retriever
        retriever = FAISSRetriever(
            embedding_model=embedding_model,
            device="cpu"  # Embeddings work fine on CPU
        )
        retriever.load_index(index_dir)

        # Initialize generator
        generator = DentalGenerator(
            model_name=model_name,
            quantize=quantize,
            device_map=device
        )

        return cls(
            retriever=retriever,
            generator=generator,
            num_docs=num_docs
        )

    def retrieve(self, question: str) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents for a question.

        Args:
            question: User question

        Returns:
            List of retrieved documents
        """
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized")

        return self.retriever.retrieve(
            query=question,
            k=self.num_docs,
            score_threshold=self.score_threshold
        )

    def generate(
        self,
        question: str,
        documents: list[RetrievedDocument],
        config: GenerationConfig | None = None
    ) -> str:
        """
        Generate answer from question and documents.

        Args:
            question: User question
            documents: Retrieved documents
            config: Generation configuration

        Returns:
            Generated answer
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized")

        # Convert documents to dict format for generator
        context_docs = [
            {
                "content": doc.content,
                "source": doc.source,
                "page_number": doc.page_number
            }
            for doc in documents
        ]

        return self.generator.generate(question, context_docs, config)

    def format_sources(self, documents: list[RetrievedDocument]) -> str:
        """
        Format sources for display.

        Args:
            documents: Retrieved documents

        Returns:
            Formatted sources string
        """
        if not documents:
            return "No sources found."

        lines = ["**Sources:**"]
        for i, doc in enumerate(documents[:3], 1):  # Show top 3
            lines.append(
                f"- {doc.source} (Page {doc.page_number}) "
                f"- Relevance: {doc.relevance_score:.2f}"
            )

        return "\n".join(lines)

    def query(
        self,
        question: str,
        config: GenerationConfig | None = None
    ) -> RAGResponse:
        """
        Full RAG query: retrieve + generate.

        Args:
            question: User question
            config: Generation configuration

        Returns:
            RAGResponse with answer and sources
        """
        # Retrieve
        documents = self.retrieve(question)

        # Generate
        answer = self.generate(question, documents, config)

        # Format sources
        formatted_sources = self.format_sources(documents)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=documents,
            formatted_sources=formatted_sources
        )

    def chat(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None
    ) -> tuple[str, list[tuple[str, str]]]:
        """
        Chat interface compatible with Gradio.

        Args:
            question: Current question
            history: Chat history (list of (user, assistant) tuples)

        Returns:
            Tuple of (response_text, updated_history)
        """
        if history is None:
            history = []

        # Get RAG response
        response = self.query(question)

        # Format full response
        full_response = f"{response.answer}\n\n{response.formatted_sources}"

        # Update history
        history.append((question, full_response))

        return full_response, history


class RAGPipelineWithHooks(RAGPipeline):
    """
    RAG Pipeline with optional pre/post processing hooks.

    Useful for adding custom logic like query rewriting or answer validation.
    """

    def __init__(
        self,
        *args,
        pre_retrieval_hook: Callable[[str], str] | None = None,
        post_generation_hook: Callable[[str, list[RetrievedDocument]], str] | None = None,
        **kwargs
    ):
        """
        Initialize pipeline with hooks.

        Args:
            pre_retrieval_hook: Function to transform query before retrieval
            post_generation_hook: Function to process answer after generation
        """
        super().__init__(*args, **kwargs)
        self.pre_retrieval_hook = pre_retrieval_hook
        self.post_generation_hook = post_generation_hook

    def query(
        self,
        question: str,
        config: GenerationConfig | None = None
    ) -> RAGResponse:
        """Query with hooks applied."""
        # Pre-retrieval hook
        processed_question = question
        if self.pre_retrieval_hook:
            processed_question = self.pre_retrieval_hook(question)

        # Retrieve with processed question
        documents = self.retrieve(processed_question)

        # Generate
        answer = self.generate(question, documents, config)  # Use original question

        # Post-generation hook
        if self.post_generation_hook:
            answer = self.post_generation_hook(answer, documents)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=documents,
            formatted_sources=self.format_sources(documents)
        )


def main():
    """Example usage of RAG pipeline."""
    index_dir = Path("data/embeddings/faiss_index")

    if not index_dir.exists():
        print(f"Index not found at {index_dir}")
        print("Run the data processing pipeline first.")
        return

    print("Initializing RAG pipeline...")

    try:
        pipeline = RAGPipeline.from_config(
            index_dir=index_dir,
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            quantize=True,
            num_docs=5
        )

        # Interactive chat
        print("\nDental Chatbot Ready! Type 'quit' to exit.\n")

        while True:
            question = input("You: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                break

            if not question:
                continue

            response = pipeline.query(question)

            print(f"\nAssistant: {response.answer}")
            print(f"\n{response.formatted_sources}\n")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires a GPU for the LLM generation.")


if __name__ == "__main__":
    main()
