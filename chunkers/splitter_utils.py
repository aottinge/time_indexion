"""
Création du text splitter avec tokenizer léger par défaut.

Le tokenizer HuggingFace Jina est lourd au démarrage ; on utilise une
approximation par caractères sauf si USE_HF_TOKENIZER ou --hf-tokenizer.
"""

from config import (
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    JINA_MODEL,
    USE_HF_TOKENIZER,
)


def estimate_tokens(text: str) -> int:
    """Estime le nombre de tokens à partir de la longueur en caractères."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def create_recursive_splitter(
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    use_hf_tokenizer: bool | None = None,
):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    use_hf = USE_HF_TOKENIZER if use_hf_tokenizer is None else use_hf_tokenizer

    if use_hf:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            JINA_MODEL,
            trust_remote_code=True,
        )
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens * CHARS_PER_TOKEN,
        chunk_overlap=chunk_overlap_tokens * CHARS_PER_TOKEN,
        length_function=len,
    )
