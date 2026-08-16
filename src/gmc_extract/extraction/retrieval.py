"""Lexical retrieval: pick the few passages relevant to a field group.

Why lexical and not embeddings/vector search. Insurance schedules use a closed, jargon-heavy
vocabulary -- "LSCS", "PED", "sub-limit", "domiciliary", "corporate floater". Exact-term
overlap is *more* precise here than semantic similarity, which happily rates
"pre-hospitalisation" and "post-hospitalisation" as near-identical when the whole task is to
tell them apart. It also avoids shipping an embedding model and a vector store to search
twenty pages.

Retrieval exists so that the LLM prompt stays small and stays inside any context window,
however long the document. The brief mentions 50-80 page policies; the samples are 3-6
pages, but chunk-and-select is what makes the difference irrelevant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from ..ingestion import PolicyDocument
from .field_specs import ALL_SPECS, FieldSpec

#: Target chunk size in characters. Big enough to keep a label with its value and any
#: trailing conditions, small enough that six chunks stay well inside a prompt budget.
_CHUNK_TARGET = 900
_CHUNK_OVERLAP_LINES = 1


@dataclass
class Snippet:
    text: str
    page: int
    score: float = 0.0


def chunk_document(document: PolicyDocument) -> List[Snippet]:
    """Split each page into paragraph-aligned chunks, never crossing a page boundary."""
    snippets: List[Snippet] = []
    for page in document.pages:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", page.search_text)
                  if block.strip()]
        current: List[str] = []
        size = 0
        for block in blocks:
            if size and size + len(block) > _CHUNK_TARGET:
                snippets.append(Snippet(text="\n\n".join(current), page=page.number))
                current = current[-_CHUNK_OVERLAP_LINES:] if _CHUNK_OVERLAP_LINES else []
                size = sum(len(part) for part in current)
            current.append(block)
            size += len(block)
        if current:
            snippets.append(Snippet(text="\n\n".join(current), page=page.number))
    return snippets


def group_cues(group: str) -> List[str]:
    """Every cue and anchor declared by the specs in a group, longest first.

    Reusing the specs means the retriever and the rule extractor share one vocabulary: any
    terminology added for the rule layer immediately improves LLM retrieval too.
    """
    cues: set = set()
    for spec in ALL_SPECS:
        if spec.group != group:
            continue
        cues.update(spec.cues)
        cues.update(spec.anchor_cues)
        cues.add(spec.label.lower())
    return sorted(cues, key=len, reverse=True)


def score_snippets(snippets: Sequence[Snippet], cues: Sequence[str]) -> List[Snippet]:
    """Score by cue overlap, weighting longer (more specific) cues higher."""
    scored: List[Snippet] = []
    for snippet in snippets:
        lowered = snippet.text.lower()
        total = 0.0
        for cue in cues:
            if cue in lowered:
                # A 30-character cue is far stronger evidence than a 3-character one.
                total += 1.0 + len(cue) / 20.0
        scored.append(Snippet(text=snippet.text, page=snippet.page, score=round(total, 2)))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def retrieve(document: PolicyDocument, group: str, limit: int = 6) -> List[Snippet]:
    """The ``limit`` most relevant snippets for a field group, in document order."""
    snippets = chunk_document(document)
    if not snippets:
        return []
    ranked = [s for s in score_snippets(snippets, group_cues(group)) if s.score > 0][:limit]
    if not ranked:
        ranked = snippets[:limit]
    ranked.sort(key=lambda item: item.page)
    return ranked


def specs_for_prompt(group: str, product_type) -> List[FieldSpec]:
    return [spec for spec in ALL_SPECS
            if spec.group == group and product_type in spec.products]
