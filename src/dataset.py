"""MuSiQue benchmark dataset loader, parser, and synthetic benchmark generator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple

from src.chunker import DeterministicChunker
from src.models import Chunk, Document, EvaluationSample


class MuSiQueDatasetParser:
    """Parses MuSiQue JSONL files or generates structured multi-hop evaluation splits."""

    def __init__(self, chunker: Optional[DeterministicChunker] = None) -> None:
        self.chunker = chunker or DeterministicChunker()

    def parse_musique_record(self, raw_record: Dict) -> Tuple[List[Document], List[Chunk], EvaluationSample]:
        """Convert a single MuSiQue JSON item into Documents, Chunks, and an EvaluationSample."""
        sample_id = str(raw_record.get("id", ""))
        question = str(raw_record.get("question", ""))
        answer = str(raw_record.get("answer", ""))
        paragraphs = raw_record.get("paragraphs", [])

        documents: List[Document] = []
        gold_chunk_ids: List[str] = []
        all_chunks: List[Chunk] = []

        for p_idx, p in enumerate(paragraphs):
            title = p.get("title", f"Doc_{p_idx}")
            p_text = p.get("paragraph_text", "")
            is_supporting = bool(p.get("is_supporting", False))

            doc = Document(
                doc_id=f"{sample_id}_p{p_idx}",
                title=title,
                text=p_text,
            )
            documents.append(doc)

            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

            if is_supporting:
                for c in chunks:
                    gold_chunk_ids.append(c.chunk_id)

        sample = EvaluationSample(
            sample_id=sample_id,
            question=question,
            gold_answer=answer,
            gold_chunk_ids=gold_chunk_ids,
            hop_count=len(raw_record.get("question_decomposition", [])) or 2,
        )

        return documents, all_chunks, sample

    def load_from_jsonl(
        self,
        filepath: str,
        max_samples: Optional[int] = None,
    ) -> Tuple[List[Document], List[Chunk], List[EvaluationSample]]:
        """Load MuSiQue validation dataset from a local JSONL file."""
        all_docs: List[Document] = []
        all_chunks: List[Chunk] = []
        samples: List[EvaluationSample] = []

        seen_chunk_ids = set()

        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if max_samples and line_idx >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                docs, chunks, sample = self.parse_musique_record(raw)
                all_docs.extend(docs)
                samples.append(sample)
                for c in chunks:
                    if c.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(c.chunk_id)
                        all_chunks.append(c)

        return all_docs, all_chunks, samples

    def load_from_hf(
        self,
        dataset_name: str = "dgslibisey/MuSiQue",
        split: str = "validation",
        max_samples: int = 10,
    ) -> Tuple[List[Document], List[Chunk], List[EvaluationSample]]:
        """Stream MuSiQue validation dataset directly from Hugging Face Hub."""
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(dataset_name, split=split, streaming=True)
        all_docs: List[Document] = []
        all_chunks: List[Chunk] = []
        samples: List[EvaluationSample] = []
        seen_chunk_ids = set()

        for idx, raw in enumerate(ds):
            if idx >= max_samples:
                break
            docs, chunks, sample = self.parse_musique_record(raw)
            all_docs.extend(docs)
            samples.append(sample)
            for c in chunks:
                if c.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(c.chunk_id)
                    all_chunks.append(c)

        return all_docs, all_chunks, samples

    @staticmethod
    def create_mock_musique_benchmark() -> Tuple[List[Document], List[EvaluationSample]]:
        """Generate verified multi-hop reasoning benchmark scenarios for testing."""
        scenarios = [
            {
                "id": "musique_hop2_001",
                "question": "Which city was the founder of Company X born in?",
                "answer": "Chicago",
                "question_decomposition": [
                    {"id": 1, "question": "Who founded Company X?"},
                    {"id": 2, "question": "Which city was John Doe born in?"},
                ],
                "paragraphs": [
                    {
                        "title": "Company X Overview",
                        "paragraph_text": "Company X is a global technology conglomerate established in 1998. John Doe founded Company X and served as its initial Chief Executive Officer for twenty years.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Biography of John Doe",
                        "paragraph_text": "John Doe is an American industrialist and engineer. John Doe was born in Chicago, Illinois, where he attended local primary schools before attending university.",
                        "is_supporting": True,
                    },
                    {
                        "title": "History of Chicago",
                        "paragraph_text": "Chicago is the most populous city in the U.S. state of Illinois. It was incorporated as a city in 1837 near a portage between the Great Lakes and the Mississippi River watershed.",
                        "is_supporting": False,
                    },
                    {
                        "title": "Tech Enterprises in California",
                        "paragraph_text": "Many software corporations operate offices in Silicon Valley. These include satellite research divisions for prominent hardware manufacturers.",
                        "is_supporting": False,
                    },
                ],
            },
            {
                "id": "musique_hop2_002",
                "question": "What award did the director of Solaris win?",
                "answer": "Grand Prix Spécial du Jury",
                "question_decomposition": [
                    {"id": 1, "question": "Who directed Solaris?"},
                    {"id": 2, "question": "What award did Andrei Tarkovsky win?"},
                ],
                "paragraphs": [
                    {
                        "title": "Solaris (film)",
                        "paragraph_text": "Solaris is a 1972 philosophical science fiction film based on Stanislaw Lem's 1961 novel of the same name. Andrei Tarkovsky directed Solaris with cinematography by Vadim Yusov.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Andrei Tarkovsky Accolades",
                        "paragraph_text": "Andrei Tarkovsky was a Soviet filmmaker and writer. Andrei Tarkovsky won Grand Prix Spécial du Jury at the 1972 Cannes Film Festival along with numerous international festival accolades.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Cannes Film Festival",
                        "paragraph_text": "The Cannes Film Festival is an annual film festival held in Cannes, France, which previews new films of all genres from around the world.",
                        "is_supporting": False,
                    },
                ],
            },
            {
                "id": "musique_hop3_003",
                "question": "Which country is the university where the inventor of Dynamo studied located in?",
                "answer": "United Kingdom",
                "question_decomposition": [
                    {"id": 1, "question": "Who invented the Dynamo?"},
                    {"id": 2, "question": "Where did Michael Faraday study?"},
                    {"id": 3, "question": "Which country is Royal Institution located in?"},
                ],
                "paragraphs": [
                    {
                        "title": "Electric Dynamo",
                        "paragraph_text": "The dynamo was the first electrical generator capable of delivering power for industry. Michael Faraday invented the Dynamo in 1831 using electromagnetic induction.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Michael Faraday Education",
                        "paragraph_text": "Michael Faraday was an English scientist who contributed to electromagnetism. Michael Faraday studied at Royal Institution as apprentice and scientific assistant.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Royal Institution",
                        "paragraph_text": "The Royal Institution is an organisation devoted to scientific education and research. Royal Institution is located in United Kingdom in the city of London.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Industrial Revolution",
                        "paragraph_text": "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States from around 1760.",
                        "is_supporting": False,
                    },
                ],
            },
        ]

        docs: List[Document] = []
        samples: List[EvaluationSample] = []
        chunker = DeterministicChunker()
        parser = MuSiQueDatasetParser(chunker)

        for s in scenarios:
            d_list, _, sample = parser.parse_musique_record(s)
            docs.extend(d_list)
            samples.append(sample)

        return docs, samples
