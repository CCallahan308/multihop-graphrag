"""Data models for Multi-Hop GraphRAG system."""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


EntityType = Literal[
    "Person",
    "Organization",
    "Location",
    "Event",
    "Work",
    "Concept",
    "Other"
]


class Entity(BaseModel):
    """Extracted or canonicalized entity."""
    name: str = Field(description="Canonical entity name")
    type: EntityType = Field(default="Concept", description="Entity category type")
    description: str = Field(default="", description="Concise summary of entity in passage")
    aliases: List[str] = Field(default_factory=list, description="Alternative surface forms")


class Relationship(BaseModel):
    """Typed relationship between two entities."""
    source_entity: str = Field(description="Source entity name")
    target_entity: str = Field(description="Target entity name")
    predicate: str = Field(description="Active relational verb or predicate type")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0, description="Extraction confidence score")
    evidence_quote: str = Field(default="", description="Supporting textual evidence quote")


class ExtractionResult(BaseModel):
    """Structured extraction output for a chunk."""
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


class Chunk(BaseModel):
    """Deterministic document chunk."""
    chunk_id: str = Field(description="Deterministic 16-character SHA-256 hash")
    text: str = Field(description="Passage text")
    doc_title: str = Field(default="", description="Parent document title or identifier")
    token_count: int = Field(default=0, description="Token length")


class Document(BaseModel):
    """Raw input document."""
    doc_id: str
    title: str
    text: str


class TraversalPath(BaseModel):
    """Relational path traversed in the property graph."""
    seed: str
    predicate_1: str
    bridge: str
    predicate_2: str
    target: str
    hop_count: int = Field(default=2, ge=1, le=3, description="Number of traversed relation edges")
    mid_chunk_id: Optional[str] = None
    target_chunk_id: Optional[str] = None


class RetrievalResult(BaseModel):
    """Result of hybrid or plain retrieval."""
    query: str
    top_chunks: List[Chunk]
    subgraph_paths: List[TraversalPath] = Field(default_factory=list)
    graph_chunk_ids: List[str] = Field(default_factory=list)
    vector_chunk_ids: List[str] = Field(default_factory=list)
    fused_scores: dict[str, float] = Field(default_factory=dict)


class SynthesisOutput(BaseModel):
    """Final answer generated from context."""
    reasoning_trace: str
    final_answer: str
    cited_chunk_ids: List[str] = Field(default_factory=list)


class EvaluationSample(BaseModel):
    """Evaluation query sample from benchmark."""
    sample_id: str
    question: str
    gold_answer: str
    gold_chunk_ids: List[str] = Field(default_factory=list)
    hop_count: int = 2


class EvaluationMetrics(BaseModel):
    """Calculated benchmark evaluation metrics."""
    total_samples: int
    multi_hop_recall: float
    mrr_at_5: float
    exact_match: float
    token_f1: float
    faithfulness: float
