"""
Découpage récursif par caractères — remplace langchain-text-splitters et transformers.
"""

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chars_for_tokens(tokens: int, chars_per_token: int = 4) -> int:
    """Approximation tokens → caractères (1 token ≈ 4 caractères en français/anglais)."""
    return max(1, tokens * chars_per_token)


def count_tokens_approx(text: str, chars_per_token: int = 4) -> int:
    return max(1, len(text) // chars_per_token) if text else 0


def recursive_split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Découpe récursivement le texte en respectant la taille et le chevauchement."""
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être inférieur à chunk_size")

    separators = separators or _DEFAULT_SEPARATORS
    return _split_recursive(text, chunk_size, chunk_overlap, separators)


def _split_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = separators[-1]
    for sep in separators:
        if sep == "" or sep in text:
            separator = sep
            break

    if separator:
        splits = text.split(separator) if separator else list(text)
    else:
        splits = list(text)

    chunks: list[str] = []
    current = ""

    for i, part in enumerate(splits):
        piece = part if not separator or i == len(splits) - 1 else part + separator
        if len(piece) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_recursive(piece, chunk_size, chunk_overlap, separators[1:]))
            continue

        candidate = current + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            current = _overlap_tail(current, chunk_overlap) + piece

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    return text[-overlap:]
