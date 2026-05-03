"""Load and index PRINCIPLES.md for structured retrieval."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Principle:
    """A single optimization principle."""

    title: str
    body: str
    applies_when: str  # extracted from **Applies when** tag
    action: str  # extracted from **Action** tag

    def to_prompt_str(self) -> str:
        """Format for LLM consumption."""
        return (
            f"### {self.title}\n"
            f"{self.body}\n"
            f"- Applies when: {self.applies_when}\n"
            f"- Action: {self.action}"
        )


def load_principles(filepath: Path) -> list[Principle]:
    """Parse a PRINCIPLES.md file into structured principles.

    Expected format:
        ## Principle Title
        Description text...
        **Applies when**: condition
        **Action**: what to do
    """
    if not filepath.exists():
        return []

    content = filepath.read_text()
    sections = re.split(r"^## ", content, flags=re.MULTILINE)

    principles = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        applies_when = _extract_tag(body, "Applies when")
        action = _extract_tag(body, "Action")

        # Clean body: remove the tag lines for the separate fields
        clean_body = re.sub(
            r"\*\*(Applies when|Action)\*\*\s*:.*$", "", body, flags=re.MULTILINE
        ).strip()

        principles.append(
            Principle(
                title=title,
                body=clean_body,
                applies_when=applies_when,
                action=action,
            )
        )

    return principles


def _extract_tag(text: str, tag_name: str) -> str:
    """Extract a **Tag**: value from markdown text."""
    pattern = rf"\*\*{tag_name}\*\*\s*:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def find_relevant_principles(
    principles: list[Principle], bottleneck_summary: str
) -> list[Principle]:
    """Find principles whose 'applies_when' matches the bottleneck.

    Simple keyword matching for now — can be upgraded to embeddings later.
    """
    bottleneck_lower = bottleneck_summary.lower()
    scored = []

    for p in principles:
        applies_lower = p.applies_when.lower()
        # Score by keyword overlap
        keywords = set(applies_lower.split())
        bottleneck_words = set(bottleneck_lower.split())
        overlap = len(keywords & bottleneck_words)
        if overlap > 0 or not p.applies_when:
            scored.append((overlap, p))

    # Sort by relevance (most overlap first), include all if none match
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored and scored[0][0] > 0:
        return [p for score, p in scored if score > 0]

    # If nothing matched, return all principles
    return principles
