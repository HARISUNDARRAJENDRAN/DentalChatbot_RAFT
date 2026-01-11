"""
Prompt Templates for Dental RAFT Chatbot
"""

# System prompt for inference
SYSTEM_PROMPT = """You are a dental education assistant helping dental students learn.
Answer questions using the provided documents.
- Cite sources using ##begin_quote## and ##end_quote## markers
- If the documents don't contain relevant information, say so clearly
- Provide detailed, educational explanations
- Focus on accuracy and cite specific facts from the documents"""


# RAFT training prompt template (Llama 3.1 Instruct format)
RAFT_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Question: {question}

Documents:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""


# Prompt for Q&A generation from chunks
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


# Prompt for answer formatting with citations
CITATION_FORMAT_PROMPT = """Format this answer with proper citations from the source text.

Original Answer:
{answer}

Source Text:
{source_text}

Rules:
1. Use ##begin_quote## and ##end_quote## to mark direct quotes
2. Only quote text that appears verbatim in the source
3. Keep quotes short and relevant (max 2 sentences each)
4. Integrate quotes naturally into the answer

Formatted Answer:"""


def format_context_documents(documents: list[dict]) -> str:
    """Format a list of documents for the prompt."""
    parts = []
    for i, doc in enumerate(documents):
        parts.append(
            f"Document {i+1} ({doc['source']}, p.{doc['page_number']}):\n{doc['content']}"
        )
    return "\n\n".join(parts)


def format_raft_training_example(
    question: str,
    context: list[dict],
    answer: str,
    system_prompt: str = SYSTEM_PROMPT
) -> str:
    """Format a complete RAFT training example."""
    context_str = format_context_documents(context)

    return RAFT_PROMPT_TEMPLATE.format(
        system_prompt=system_prompt,
        question=question,
        context=context_str,
        answer=answer
    )


def format_inference_prompt(
    question: str,
    context: list[dict],
    system_prompt: str = SYSTEM_PROMPT
) -> str:
    """Format prompt for inference (no answer)."""
    context_str = format_context_documents(context)

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Question: {question}

Documents:
{context_str}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

"""
    return prompt
