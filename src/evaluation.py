"""Evaluation metrics and harness for multi-hop RAG systems."""

from __future__ import annotations

import re
import string
from typing import Dict, List, Sequence, Set
from src.models import EvaluationMetrics, EvaluationSample, RetrievalResult, SynthesisOutput


class MetricCalculator:
    """Computes exact match, token F1, multi-hop recall, MRR@5, and factual grounding."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Official SQuAD / MuSiQue answer text normalization."""
        if not text:
            return ""

        # Lowercase
        s = str(text).lower()

        # Remove punctuation
        exclude = set(string.punctuation)
        s = "".join(ch for ch in s if ch not in exclude)

        # Remove articles ('a', 'an', 'the')
        s = re.sub(r"\b(a|an|the)\b", " ", s)

        # Collapse whitespace
        return " ".join(s.split())

    @classmethod
    def _single_token_f1(cls, prediction: str, gold: str) -> float:
        """Token-level F1 score between prediction and single ground truth string."""
        pred_tokens = cls.normalize_text(prediction).split()
        gold_tokens = cls.normalize_text(gold).split()

        if not pred_tokens or not gold_tokens:
            return 1.0 if pred_tokens == gold_tokens else 0.0

        common_tokens: Dict[str, int] = {}
        for token in pred_tokens:
            if token in gold_tokens:
                common_tokens[token] = common_tokens.get(token, 0) + 1

        same_token_count = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(common_tokens.keys()))
        if same_token_count == 0:
            return 0.0

        precision = same_token_count / len(pred_tokens)
        recall = same_token_count / len(gold_tokens)
        f1 = (2.0 * precision * recall) / (precision + recall)
        return float(f1)

    @classmethod
    def compute_exact_match(cls, prediction: str, gold: str, aliases: Sequence[str] = ()) -> float:
        """Exact string match after official text normalization across gold and aliases."""
        norm_pred = cls.normalize_text(prediction)
        if not norm_pred:
            return 0.0

        candidates = [gold] + list(aliases)
        for target in candidates:
            if norm_pred == cls.normalize_text(target):
                return 1.0
        return 0.0

    @classmethod
    def compute_token_f1(cls, prediction: str, gold: str, aliases: Sequence[str] = ()) -> float:
        """Token-level F1 score taking maximum across gold answer and aliases."""
        candidates = [gold] + list(aliases)
        max_f1 = 0.0
        for target in candidates:
            score = cls._single_token_f1(prediction, target)
            if score > max_f1:
                max_f1 = score
        return max_f1

    @staticmethod
    def compute_multi_hop_recall(retrieved_chunk_ids: Sequence[str], gold_chunk_ids: Sequence[str]) -> float:
        """Return 1.0 if ALL gold chunks are present in retrieved set, else 0.0."""
        if not gold_chunk_ids:
            return 1.0
        retrieved_set = set(retrieved_chunk_ids)
        all_present = all(gid in retrieved_set for gid in gold_chunk_ids)
        return 1.0 if all_present else 0.0

    @staticmethod
    def compute_mrr_at_k(retrieved_chunk_ids: Sequence[str], gold_chunk_ids: Sequence[str], k: int = 5) -> float:
        """Mean Reciprocal Rank within top-k ranks for the first relevant gold chunk."""
        if not gold_chunk_ids:
            return 1.0
        gold_set = set(gold_chunk_ids)
        for rank, cid in enumerate(retrieved_chunk_ids[:k], start=1):
            if cid in gold_set:
                return 1.0 / rank
        return 0.0

    @classmethod
    def compute_faithfulness(cls, generated_text: str, context_text: str) -> float:
        """Estimate factual grounding by measuring claim token grounding against context."""
        claims = [c.strip() for c in re.split(r"[.!?]+", generated_text) if len(c.strip()) > 5]
        if not claims:
            return 1.0

        norm_context = cls.normalize_text(context_text)
        context_tokens = set(norm_context.split())

        grounded_claims = 0
        for claim in claims:
            claim_tokens = set(cls.normalize_text(claim).split())
            if not claim_tokens:
                continue
            # Claim is considered grounded if >= 70% of content tokens appear in context
            overlap = len(claim_tokens & context_tokens)
            if (overlap / len(claim_tokens)) >= 0.70:
                grounded_claims += 1

        return float(grounded_claims / len(claims))


class EvaluationHarness:
    """Executes evaluation benchmarks comparing GraphRAG against Plain RAG."""

    def __init__(self) -> None:
        self.calculator = MetricCalculator()

    def evaluate_system(
        self,
        samples: Sequence[EvaluationSample],
        retrieval_fn,
        generation_fn,
        on_sample_evaluated: Optional[Callable[[Dict], None]] = None,
    ) -> Tuple[EvaluationMetrics, List[Dict]]:
        """Run evaluation over a test corpus with per-sample callback support."""
        total = len(samples)
        if total == 0:
            return (
                EvaluationMetrics(
                    total_samples=0,
                    multi_hop_recall=0.0,
                    mrr_at_5=0.0,
                    exact_match=0.0,
                    token_f1=0.0,
                    faithfulness=0.0,
                ),
                [],
            )

        recalls: List[float] = []
        mrrs: List[float] = []
        ems: List[float] = []
        f1s: List[float] = []
        faithfulness_scores: List[float] = []
        detailed_records: List[Dict] = []

        for sample in samples:
            # 1. Retrieve context
            retrieval_res: RetrievalResult = retrieval_fn(sample.question)
            retrieved_ids = [c.chunk_id for c in retrieval_res.top_chunks]

            # 2. Compute retrieval metrics
            rec = self.calculator.compute_multi_hop_recall(retrieved_ids, sample.gold_chunk_ids)
            mrr = self.calculator.compute_mrr_at_k(retrieved_ids, sample.gold_chunk_ids, k=5)
            recalls.append(rec)
            mrrs.append(mrr)

            # 3. Generate answer
            context_text = "\n".join(c.text for c in retrieval_res.top_chunks)
            try:
                gen_res: SynthesisOutput = generation_fn(sample.question, context_text, retrieval_res)
            except TypeError:
                gen_res = generation_fn(sample.question, context_text)

            # 4. Compute synthesis metrics with alias support
            aliases = getattr(sample, "answer_aliases", ())
            em = self.calculator.compute_exact_match(gen_res.final_answer, sample.gold_answer, aliases=aliases)
            f1 = self.calculator.compute_token_f1(gen_res.final_answer, sample.gold_answer, aliases=aliases)
            faith = self.calculator.compute_faithfulness(gen_res.final_answer, context_text)

            ems.append(em)
            f1s.append(f1)
            faithfulness_scores.append(faith)

            record = {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "gold_answer": sample.gold_answer,
                "predicted_answer": gen_res.final_answer,
                "gold_chunks": sample.gold_chunk_ids,
                "retrieved_chunks": retrieved_ids,
                "multi_hop_recall": rec,
                "mrr_at_5": mrr,
                "exact_match": em,
                "token_f1": f1,
                "faithfulness": faith,
            }
            detailed_records.append(record)
            if on_sample_evaluated:
                on_sample_evaluated(record)

        avg_metrics = EvaluationMetrics(
            total_samples=total,
            multi_hop_recall=float(sum(recalls) / total),
            mrr_at_5=float(sum(mrrs) / total),
            exact_match=float(sum(ems) / total),
            token_f1=float(sum(f1s) / total),
            faithfulness=float(sum(faithfulness_scores) / total),
        )

        return avg_metrics, detailed_records
