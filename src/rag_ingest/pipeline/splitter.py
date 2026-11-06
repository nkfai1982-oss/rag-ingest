"""Recursive character text splitter."""


class RecursiveTextSplitter:
    """Splits text recursively using hierarchical separators with configurable size and overlap."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks of at most chunk_size characters with overlap."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []

        # Find the first separator present in the text
        separator = separators[-1]
        remaining_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1 :]
                break

        # Split text by the chosen separator
        splits: list[str] = text.split(separator) if separator != "" else list(text)

        good_splits: list[str] = []
        for s in splits:
            if not s.strip():
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if remaining_separators:
                    sub_chunks = self._split(s, remaining_separators)
                    good_splits.extend(sub_chunks)
                else:
                    # Character slice fallback
                    for idx in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        good_splits.append(s[idx : idx + self.chunk_size])

        # Merge splits into chunks respecting chunk_size and chunk_overlap
        current_chunk: list[str] = []
        current_length = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk and separator else 0)
            if current_length + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_length += piece_len
            else:
                if current_chunk:
                    merged = separator.join(current_chunk).strip()
                    if merged:
                        final_chunks.append(merged)

                    # Build overlap window from tail of current_chunk
                    overlap_chunk: list[str] = []
                    overlap_len = 0
                    for prev_piece in reversed(current_chunk):
                        item_len = len(prev_piece) + (len(separator) if overlap_chunk else 0)
                        if overlap_len + item_len <= self.chunk_overlap:
                            overlap_chunk.insert(0, prev_piece)
                            overlap_len += item_len
                        else:
                            break

                    current_chunk = overlap_chunk
                    current_length = sum(len(p) for p in current_chunk) + max(
                        0, (len(current_chunk) - 1) * len(separator)
                    )

                current_chunk.append(piece)
                current_length += len(piece) + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            merged = separator.join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks
