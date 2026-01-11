"""
Dental Education Chatbot - Gradio App for HuggingFace ZeroGPU Spaces

This app provides a chat interface for the RAFT-trained dental chatbot.
Optimized for deployment on HuggingFace Spaces with ZeroGPU.
"""

import gradio as gr
import torch
import pickle
from pathlib import Path

# ZeroGPU support
try:
    import spaces
    ZEROGPU_AVAILABLE = True
except ImportError:
    ZEROGPU_AVAILABLE = False
    # Mock decorator for local testing
    class spaces:
        @staticmethod
        def GPU(duration=60):
            def decorator(func):
                return func
            return decorator

import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Configuration
MODEL_NAME = "your-username/llama-3.1-8b-dental-raft"  # Change to your fine-tuned model
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
INDEX_DIR = "faiss_index"
NUM_DOCS = 5

# Global variables (loaded once at startup)
embed_model = None
faiss_index = None
chunks = None
llm_model = None
tokenizer = None


def load_embedding_model():
    """Load embedding model (CPU)."""
    global embed_model
    if embed_model is None:
        print("Loading embedding model...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return embed_model


def load_faiss_index():
    """Load FAISS index and chunks (CPU)."""
    global faiss_index, chunks

    if faiss_index is None:
        index_path = Path(INDEX_DIR) / "index.faiss"
        chunks_path = Path(INDEX_DIR) / "chunks.pkl"

        if index_path.exists() and chunks_path.exists():
            print("Loading FAISS index...")
            faiss_index = faiss.read_index(str(index_path))

            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)

            print(f"Loaded {faiss_index.ntotal} vectors")
        else:
            print(f"Warning: Index not found at {INDEX_DIR}")
            faiss_index = None
            chunks = []

    return faiss_index, chunks


@spaces.GPU(duration=120)
def load_llm():
    """Load LLM model (GPU)."""
    global llm_model, tokenizer

    if llm_model is None:
        print("Loading LLM model...")

        llm_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("LLM loaded successfully")

    return llm_model, tokenizer


def retrieve_context(question: str, k: int = NUM_DOCS) -> list[dict]:
    """Retrieve relevant documents using FAISS (CPU operation)."""
    embed_model = load_embedding_model()
    index, chunk_data = load_faiss_index()

    if index is None or not chunk_data:
        return []

    # Embed question
    query_embedding = embed_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    # Search
    scores, indices = index.search(query_embedding, k)

    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(chunk_data):
            chunk = chunk_data[idx]
            results.append({
                "content": chunk["text"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "relevance_score": float(score)
            })

    return results


@spaces.GPU(duration=60)
def generate_answer(question: str, context_docs: list[dict]) -> str:
    """Generate answer using LLM (GPU operation)."""
    model, tok = load_llm()

    # Format context
    context_str = "\n\n".join([
        f"Document {i+1} ({doc['source']}, p.{doc['page_number']}):\n{doc['content']}"
        for i, doc in enumerate(context_docs)
    ])

    # Llama 3.1 Instruct format
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a dental education assistant. Answer questions using the provided documents.
- Cite sources using ##begin_quote## and ##end_quote## markers
- If documents don't contain relevant information, say so clearly
- Provide detailed, educational explanations<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Question: {question}

Documents:
{context_str}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

"""

    # Tokenize
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id
        )

    # Decode
    response = tok.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        answer = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    else:
        answer = response[len(prompt):] if len(response) > len(prompt) else response

    # Clear GPU cache
    torch.cuda.empty_cache()

    return answer.strip()


def format_sources(context_docs: list[dict]) -> str:
    """Format sources for display."""
    if not context_docs:
        return ""

    lines = ["\n\n**Sources:**"]
    for doc in context_docs[:3]:
        lines.append(
            f"- {doc['source']} (Page {doc['page_number']}) "
            f"- Relevance: {doc['relevance_score']:.2f}"
        )

    return "\n".join(lines)


def chat(message: str, history: list[dict]) -> str:
    """
    Main chat function.

    Args:
        message: User message
        history: Chat history

    Returns:
        Assistant response
    """
    if not message.strip():
        return "Please enter a question about dental topics."

    try:
        # Step 1: Retrieve context (CPU)
        context_docs = retrieve_context(message, k=NUM_DOCS)

        if not context_docs:
            return "I couldn't find relevant information in the dental textbooks. Please try rephrasing your question."

        # Step 2: Generate answer (GPU)
        answer = generate_answer(message, context_docs)

        # Step 3: Add sources
        sources = format_sources(context_docs)
        full_response = answer + sources

        return full_response

    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again."


# Example questions
EXAMPLES = [
    "What are the indications for root canal therapy?",
    "Explain the stages of tooth development.",
    "What are the contraindications for dental implants?",
    "Describe the treatment protocol for periapical abscess.",
    "What are the different types of dental caries classifications?",
]

# Custom CSS
CSS = """
.gradio-container {
    max-width: 900px !important;
}
footer {
    display: none !important;
}
"""

# Build Gradio interface
with gr.Blocks(css=CSS, title="Dental Education Chatbot") as demo:
    gr.Markdown(
        """
        # 🦷 Dental Education Chatbot

        Ask questions about dental topics! This chatbot is trained on 100+ dental textbooks
        using RAFT (Retrieval Augmented Fine-Tuning) methodology.

        **Features:**
        - Answers based on authoritative dental textbooks
        - Citations with source references
        - Optimized for dental students

        *Powered by Llama 3.1 8B + FAISS + PubMedBERT*
        """
    )

    chatbot = gr.ChatInterface(
        fn=chat,
        examples=EXAMPLES,
        title="",
        description="",
        theme="soft",
        retry_btn="🔄 Retry",
        undo_btn="↩️ Undo",
        clear_btn="🗑️ Clear",
    )

    gr.Markdown(
        """
        ---
        **Disclaimer:** This chatbot is for educational purposes only.
        Always consult with qualified dental professionals for clinical decisions.

        🤖 Built with RAFT methodology | [Learn more about RAFT](https://gorilla.cs.berkeley.edu/blogs/9_raft.html)
        """
    )

# Launch
if __name__ == "__main__":
    # Initialize models at startup
    print("Initializing models...")
    load_embedding_model()
    load_faiss_index()

    # Launch app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
