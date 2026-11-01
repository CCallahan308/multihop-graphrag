"""Entity and relationship extraction and canonicalization engine."""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.embeddings import BaseEmbeddingProvider
from src.models import Chunk, Entity, ExtractionResult, Relationship

logger = logging.getLogger(__name__)


class EntityCanonicalizer:
    """Canonicalizes entities using lexical normalization, fuzzy token matching, and dense vector matching."""

    ABBREVIATION_MAP = {
        "inc": "incorporated",
        "corp": "corporation",
        "co": "company",
        "ltd": "limited",
        "univ": "university",
        "st": "saint",
        "dept": "department",
    }

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        similarity_threshold: float = 0.88,
        fuzzy_threshold: float = 0.85,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold

    @classmethod
    def normalize_name(cls, name: str) -> str:
        """Strip whitespace, collapse consecutive spaces, and strip surrounding quotes."""
        cleaned = re.sub(r"\s+", " ", name.strip())
        return cleaned.strip("\"' ")

    @classmethod
    def compute_fuzzy_token_similarity(cls, name_a: str, name_b: str) -> float:
        """Compute fuzzy token similarity using expanded abbreviation Jaccard overlap."""
        norm_a = cls.normalize_name(name_a).lower()
        norm_b = cls.normalize_name(name_b).lower()

        if norm_a == norm_b:
            return 1.0

        tokens_a = [cls.ABBREVIATION_MAP.get(t, t) for t in re.findall(r"\w+", norm_a)]
        tokens_b = [cls.ABBREVIATION_MAP.get(t, t) for t in re.findall(r"\w+", norm_b)]

        if not tokens_a or not tokens_b:
            return 0.0

        set_a = set(tokens_a)
        set_b = set(tokens_b)

        # Exact match after expansion
        if set_a == set_b:
            return 1.0

        # Substring / subset containment for acronyms or titles
        if set_a.issubset(set_b) or set_b.issubset(set_a):
            shorter = min(len(set_a), len(set_b))
            longer = max(len(set_a), len(set_b))
            if shorter >= 1 and (shorter / longer) >= 0.5:
                return 0.90

        # Jaccard overlap
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return float(intersection / union) if union > 0 else 0.0

    def canonicalize(self, entities: Sequence[Entity]) -> List[Entity]:
        """Merge duplicate or highly similar entities into single canonical nodes with aliases."""
        if not entities:
            return []

        # 1. Exact lexical grouping via case-folded keys
        lexical_groups: Dict[str, List[Entity]] = {}
        for ent in entities:
            norm_name = self.normalize_name(ent.name)
            if not norm_name:
                continue
            key = norm_name.lower()
            if key not in lexical_groups:
                lexical_groups[key] = []
            lexical_groups[key].append(ent)

        # Build initial deduplicated entity representations
        initial_merged: List[Entity] = []
        for key, group in lexical_groups.items():
            primary = group[0]
            canonical_name = max((e.name for e in group), key=len)
            aliases: Set[str] = set()
            descriptions: List[str] = []

            for e in group:
                norm_e = self.normalize_name(e.name)
                if norm_e.lower() != canonical_name.lower():
                    aliases.add(norm_e)
                aliases.update(e.aliases)
                if e.description and e.description not in descriptions:
                    descriptions.append(e.description)

            ent_type = primary.type
            merged_desc = "; ".join(descriptions) if descriptions else primary.description

            initial_merged.append(
                Entity(
                    name=canonical_name,
                    type=ent_type,
                    description=merged_desc,
                    aliases=sorted(list(aliases)),
                )
            )

        if len(initial_merged) <= 1:
            return initial_merged

        # 2. Dense vector embedding matching across entity names
        names = [e.name for e in initial_merged]
        embeddings = self.embedding_provider.embed_texts(names)

        # 3. Disjoint-set clustering with semantic vector or fuzzy token threshold
        parent = list(range(len(initial_merged)))

        def find(i: int) -> int:
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        num_entities = len(initial_merged)
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                # Verify entity types are compatible before merge
                if initial_merged[i].type != initial_merged[j].type and initial_merged[i].type != "Concept" and initial_merged[j].type != "Concept":
                    continue

                # Fuzzy token check
                fuzzy_sim = self.compute_fuzzy_token_similarity(names[i], names[j])

                # Vector cosine check
                sim = self.embedding_provider.cosine_similarity(embeddings[i], embeddings[j])

                if sim >= self.similarity_threshold or fuzzy_sim >= self.fuzzy_threshold:
                    desc_i = initial_merged[i].description.lower()
                    desc_j = initial_merged[j].description.lower()

                    if not desc_i or not desc_j or self._descriptions_compatible(desc_i, desc_j):
                        union(i, j)

        # 4. Group by root and merge into canonical nodes
        clusters: Dict[int, List[Entity]] = {}
        for i in range(num_entities):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(initial_merged[i])

        final_canonical: List[Entity] = []
        for cluster in clusters.values():
            primary = min(cluster, key=lambda e: len(e.name))
            all_aliases: Set[str] = set()
            all_descriptions: List[str] = []

            for e in cluster:
                if e.name != primary.name:
                    all_aliases.add(e.name)
                all_aliases.update(e.aliases)
                if e.description and e.description not in all_descriptions:
                    all_descriptions.append(e.description)

            final_canonical.append(
                Entity(
                    name=primary.name,
                    type=primary.type,
                    description="; ".join(all_descriptions),
                    aliases=sorted(list(all_aliases)),
                )
            )

        return final_canonical

    @staticmethod
    def _descriptions_compatible(desc_a: str, desc_b: str) -> bool:
        """Check for token overlap between two descriptions."""
        tokens_a = set(re.findall(r"\w+", desc_a))
        tokens_b = set(re.findall(r"\w+", desc_b))
        if not tokens_a or not tokens_b:
            return True
        overlap = len(tokens_a & tokens_b)
        min_len = min(len(tokens_a), len(tokens_b))
        return (overlap / min_len) >= 0.20


FEW_SHOT_EXTRACTION_PROMPT = """You are an expert knowledge graph extraction engine.
Extract entities and relationships from the text below according to the exact JSON schema.

Few-Shot Examples:
Example 1:
Passage: "Grant Green featuring performances by Green recorded and released on Blue Note label in 1961."
JSON Output:
{
  "entities": [
    {"name": "Grant Green", "type": "Person", "description": "American jazz guitarist"},
    {"name": "Green", "type": "Person", "description": "Short form of Grant Green"},
    {"name": "Blue Note", "type": "Organization", "description": "Record label"}
  ],
  "relationships": [
    {"source_entity": "Grant Green", "target_entity": "Blue Note", "predicate": "RECORDED_ON", "confidence": 0.95, "evidence_quote": "performances by Green recorded and released on Blue Note label"},
    {"source_entity": "Green", "target_entity": "Blue Note", "predicate": "RECORDED_ON", "confidence": 0.95, "evidence_quote": "performances by Green recorded and released on Blue Note label"}
  ]
}

Example 2:
Passage: "Steve Hillage is an English musician. Steve Hillage married Miquette Giraudy in 1978. Hillage was the lead performer of the album Green."
JSON Output:
{
  "entities": [
    {"name": "Steve Hillage", "type": "Person", "description": "English musician"},
    {"name": "Miquette Giraudy", "type": "Person", "description": "French synthesizer player and spouse"},
    {"name": "Green", "type": "Work", "description": "1978 studio album"}
  ],
  "relationships": [
    {"source_entity": "Steve Hillage", "target_entity": "Miquette Giraudy", "predicate": "SPOUSE_OF", "confidence": 0.98, "evidence_quote": "Steve Hillage married Miquette Giraudy in 1978"},
    {"source_entity": "Miquette Giraudy", "target_entity": "Steve Hillage", "predicate": "SPOUSE_OF", "confidence": 0.98, "evidence_quote": "Steve Hillage married Miquette Giraudy in 1978"},
    {"source_entity": "Steve Hillage", "target_entity": "Green", "predicate": "PERFORMER_OF", "confidence": 0.95, "evidence_quote": "Hillage was the lead performer of the album Green"}
  ]
}

Passage to Extract:
\"\"\"{text}\"\"\"

Provide strictly valid JSON adhering to the schema without introductory or markdown conversational text.
"""


class LLMEntityRelationshipExtractor:
    """Structured LLM extractor using few-shot prompt with automatic fallback."""

    def __init__(
        self,
        llm_callable: Optional[Callable[[str], str]] = None,
        fallback_extractor: Optional[RuleBasedExtractor] = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.fallback_extractor = fallback_extractor or RuleBasedExtractor()

    def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities and relations using LLM, with fallback on failure."""
        if not self.llm_callable:
            return self.fallback_extractor.extract(chunk)

        prompt = FEW_SHOT_EXTRACTION_PROMPT.format(text=chunk.text)
        try:
            raw_response = self.llm_callable(prompt)
            # Find JSON block
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                payload = json.loads(json_match.group(0))
                entities = [Entity(**e) for e in payload.get("entities", [])]
                relationships = [Relationship(**r) for r in payload.get("relationships", [])]
                if entities or relationships:
                    return ExtractionResult(entities=entities, relationships=relationships)
        except Exception as e:
            logger.warning(f"LLM extraction error: {e}, using fallback extractor.")

        return self.fallback_extractor.extract(chunk)


class RuleBasedExtractor:
    """Heuristic extractor for entity and relation extraction without external API dependencies.

    Extracts capitalized noun entities, relational phrases, and bi-directional edges.
    Serves as safe fallback or offline test engine.
    """

    def extract(self, chunk: Chunk) -> ExtractionResult:
        text = chunk.text
        entities: List[Entity] = []
        relationships: List[Relationship] = []

        # Find capitalized entity sequences with unicode and lowercase particle support
        pattern = r"\b[A-Z\u00C0-\u024F][\w\u00C0-\u024F]*(?:\s+(?:du|de|la|le|of|the|and|in|at|van|von|da|di)\s+[A-Z\u00C0-\u024F][\w\u00C0-\u024F]*|\s+[A-Z\u00C0-\u024F][\w\u00C0-\u024F]*)*\b"
        matches = list(re.finditer(pattern, text))
        seen_names: Set[str] = set()

        # Stop words to ignore as standalone entities
        stop_entities = {
            "The", "A", "An", "In", "On", "At", "By", "For", "With",
            "However", "Therefore", "It", "They", "He", "She", "This",
            "That", "These", "Those", "After", "Before", "During", "Earlier"
        }

        for m in matches:
            name = m.group(0).strip()
            # Normalize leading articles (e.g., 'The Royal Institution' -> 'Royal Institution')
            if name.startswith(("The ", "A ", "An ")):
                prefix_len = 4 if name.startswith("The ") else (2 if name.startswith("A ") else 3)
                remainder = name[prefix_len:].strip()
                if remainder and remainder[0].isupper() and remainder not in stop_entities:
                    name = remainder
            if name in stop_entities or len(name) <= 1:
                continue
            if name.lower() not in seen_names:
                seen_names.add(name.lower())
                # Type inference based on surface form
                ent_type = "Concept"
                if re.search(r"\b(City|State|Country|River|Island|Portage|Watershed)\b", name):
                    ent_type = "Location"
                elif re.search(r"\b(Inc|Corp|LLC|Company|Institution|University|Festival|Record|Records)\b", name):
                    ent_type = "Organization"
                entities.append(
                    Entity(
                        name=name,
                        type=ent_type,
                        description=f"Entity mentioned in {chunk.doc_title or 'passage'}",
                    )
                )

        # Relational triple extraction with bidirectional typing
        sentences = re.split(r"[.!?]+", text)
        rel_patterns = [
            (r"\b(?:was\s+born\s+in|born\s+in)\b", "BORN_IN", "BIRTHPLACE_OF"),
            (r"\b(?:founded|established|created)\b", "FOUNDED", "FOUNDED_BY"),
            (r"\b(?:married\s+to|married|spouse\s+of)\b", "SPOUSE_OF", "SPOUSE_OF"),
            (r"\b(?:directed|directed\s+by)\b", "DIRECTED", "DIRECTED_BY"),
            (r"\b(?:won|awarded|recipient\s+of)\b", "WON", "AWARDED_TO"),
            (r"\b(?:located\s+in|situated\s+in)\b", "LOCATED_IN", "LOCATION_OF"),
            (r"\b(?:invented|developed)\b", "INVENTED", "INVENTED_BY"),
            (r"\b(?:studied\s+at|apprentice\s+at|attended)\b", "STUDIED_AT", "AFFILIATION_OF"),
            (r"\b(?:performer\s+of|lead\s+performer\s+of|featuring\s+performances\s+by)\b", "PERFORMER_OF", "PERFORMER"),
            (r"\b(?:recorded\s+and\s+released\s+on|recorded\s+on|released\s+on)\b", "RECORDED_ON", "RELEASED"),
        ]

        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue

            found_entities = sorted(
                (e.name for e in entities if e.name in sent_str),
                key=sent_str.find,
            )
            if len(found_entities) >= 2:
                for i in range(len(found_entities) - 1):
                    src = found_entities[i]
                    tgt = found_entities[i + 1]
                    if src.lower() == tgt.lower():
                        continue

                    # Search text between entities for predicates
                    sub_pat = re.escape(src) + r"\s+(.*?)\s+" + re.escape(tgt)
                    match_rel = re.search(sub_pat, sent_str)
                    forward_pred = "RELATED_TO"
                    reverse_pred = "RELATED_TO"

                    if match_rel:
                        between = match_rel.group(1).strip()
                        matched_custom = False
                        for pat, f_pred, r_pred in rel_patterns:
                            if re.search(pat, between, re.IGNORECASE):
                                forward_pred = f_pred
                                reverse_pred = r_pred
                                matched_custom = True
                                break
                        if not matched_custom and len(between.split()) <= 4:
                            cleaned_pred = re.sub(r"[^\w\s]", "", between).upper().replace(" ", "_")
                            if cleaned_pred:
                                forward_pred = cleaned_pred
                                reverse_pred = f"INVERSE_{cleaned_pred}"

                    # Add forward and reverse relationship
                    relationships.append(
                        Relationship(
                            source_entity=src,
                            target_entity=tgt,
                            predicate=forward_pred,
                            confidence=0.90,
                            evidence_quote=sent_str[:200],
                        )
                    )
                    relationships.append(
                        Relationship(
                            source_entity=tgt,
                            target_entity=src,
                            predicate=reverse_pred,
                            confidence=0.88,
                            evidence_quote=sent_str[:200],
                        )
                    )

        return ExtractionResult(entities=entities, relationships=relationships)
