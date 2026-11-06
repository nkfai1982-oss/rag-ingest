"""Text sanitization and cleaning processor."""

import re


class CleanProcessor:
    """Sanitizes text by stripping control characters and excessive newlines.

    Preserves markdown table and code block structures.
    """

    # Regex matching control characters except tab and newline
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    # Regex matching multiple empty lines
    _EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")
    # Regex matching multiple horizontal spaces
    _HORIZONTAL_SPACES_RE = re.compile(r"[^\S\r\n]{2,}")

    def clean(self, text: str) -> str:
        """Sanitizes text content."""
        if not text:
            return ""

        # 1. Strip non-printable ASCII control characters
        cleaned = self._CONTROL_CHAR_RE.sub("", text)

        # 2. Normalize Windows/Mac line endings to standard Unix \n
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Process line by line to preserve markdown tables and code blocks
        lines = cleaned.split("\n")
        processed_lines: list[str] = []
        in_code_block = False

        for line in lines:
            trimmed_line = line.strip()
            if trimmed_line.startswith("```"):
                in_code_block = not in_code_block
                processed_lines.append(line)
                continue

            if in_code_block:
                processed_lines.append(line)
            else:
                # Outside code blocks, collapse excessive spaces unless in table row
                if "|" in line:
                    processed_lines.append(line.rstrip())
                else:
                    line_normalized = self._HORIZONTAL_SPACES_RE.sub(" ", line)
                    processed_lines.append(line_normalized.rstrip())

        result = "\n".join(processed_lines)

        # 4. Collapse 3+ newlines into 2 (paragraph break)
        result = self._EXCESSIVE_NEWLINES_RE.sub("\n\n", result)

        return result.strip()
