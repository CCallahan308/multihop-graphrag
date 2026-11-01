"""Run a comparative GraphRAG evaluation and export inspectable results."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

# Keep direct script invocation available from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def positive_integer(value: str) -> int:
    """Reject missing, zero, and negative evaluation limits at the CLI boundary."""
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if count <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GraphRAG with plain dense retrieval.")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--data-file", type=Path, help="A local MuSiQue JSONL file")
    sources.add_argument("--use-hf", action="store_true", help="Stream dgslibisey/MuSiQue validation records")
    parser.add_argument("--max-samples", type=positive_integer, default=10, help="Maximum questions; built-in smoke test has three")
    parser.add_argument("--storage-dir", type=Path, default=Path("./data_store"), help="Graph and vector stores; existing contents are reset")
    parser.add_argument("--checkpoint-file", type=Path, default=Path("./data_store/eval_checkpoint.json"))
    parser.add_argument("--log-file", type=Path, default=Path("./data_store/eval_errors.jsonl"))
    parser.add_argument("--report-file", type=Path, default=Path("./data_store/report.html"), help="Standalone HTML evaluation report")
    parser.add_argument("--results-file", type=Path, default=Path("./data_store/results.json"), help="Versioned JSON results for both systems")
    parser.add_argument("--use-sentence-transformers", action="store_true", help="Use BAAI/bge-small-en-v1.5 instead of hash projection")
    parser.add_argument("--quiet", action="store_true", help="Hide progress messages; retain results and errors")
    args = parser.parse_args(argv)
    if args.data_file is not None and not args.data_file.is_file():
        parser.error(f"--data-file is not a readable file: {args.data_file}")

    output_paths = [args.checkpoint_file, args.log_file, args.report_file, args.results_file]
    resolved_outputs = [path.resolve() for path in output_paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        parser.error("checkpoint, log, report, and results paths must be different")
    if args.data_file is not None:
        source_path = args.data_file.resolve()
        storage_path = args.storage_dir.resolve()
        if source_path == storage_path or storage_path in source_path.parents:
            parser.error("--data-file must be outside --storage-dir, whose contents are reset")
        if source_path in resolved_outputs:
            parser.error("output paths must not overwrite --data-file")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Invalid arguments fail before database initialization or optional imports.
    from src.dataset import MuSiQueDatasetParser
    from src.embeddings import HashProjectionEmbedding, SentenceTransformerEmbedding
    from src.pipeline import MultiHopGraphRAGSystem
    from src.report_html import render_html_report
    from src.reporting import build_report, render_console

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    progress("Loading evaluation questions...")
    dataset_parser = MuSiQueDatasetParser()
    if args.use_hf:
        dataset_kind = "huggingface"
        dataset_label = "dgslibisey/MuSiQue validation"
        docs, _, samples = dataset_parser.load_from_hf(max_samples=args.max_samples)
    elif args.data_file is not None:
        dataset_kind = "jsonl"
        dataset_label = args.data_file.name
        docs, _, samples = dataset_parser.load_from_jsonl(str(args.data_file), max_samples=args.max_samples)
    else:
        dataset_kind = "synthetic"
        dataset_label = "Built-in synthetic smoke test"
        docs, samples = MuSiQueDatasetParser.create_mock_musique_benchmark()
        samples = samples[:args.max_samples]

    if args.use_sentence_transformers:
        progress("Loading the optional sentence-transformer embedding provider...")
        embedder = SentenceTransformerEmbedding(model_name="BAAI/bge-small-en-v1.5")
        if embedder._fallback_provider is not None:
            embedding_label = "Hash projection (sentence-transformer fallback)"
        else:
            embedding_label = "BAAI/bge-small-en-v1.5"
    else:
        embedder = HashProjectionEmbedding(dimension=384)
        embedding_label = "Hash projection, 384 dimensions"

    progress(f"Indexing {len(docs)} documents in Kuzu and LanceDB...")
    system = MultiHopGraphRAGSystem(
        storage_dir=str(args.storage_dir), embedding_provider=embedder, clean_start=True,
    )
    all_chunks = system.ingest_documents(docs)
    system.extract_and_index_knowledge(all_chunks)

    checkpoint_path = args.checkpoint_file.resolve()
    log_path = args.log_file.resolve()
    report_path = args.report_file.resolve()
    results_path = args.results_file.resolve()
    for path in (checkpoint_path, log_path, report_path, results_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    # Start a fresh log even when there are no evaluated questions.
    log_path.write_text("", encoding="utf-8")
    checkpoint_records: list[dict] = []

    def checkpoint_callback(record: dict) -> None:
        checkpoint_records.append(record)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint_path.write_text(json.dumps({
            "completed_samples": len(checkpoint_records),
            "total_samples": len(samples),
            "records": checkpoint_records,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    checkpoint_path.write_text(json.dumps({
        "completed_samples": 0, "total_samples": len(samples), "records": [],
    }, indent=2) + "\n", encoding="utf-8")
    progress(f"Evaluating {len(samples)} questions...")
    results = system.run_benchmark(samples, on_sample_evaluated=checkpoint_callback)
    if args.use_sentence_transformers and embedder._fallback_provider is not None:
        # Encoding can fail after a model loads; expose that degraded run too.
        embedding_label = (
            f"Hash projection fallback, {embedder.dimension} dimensions"
            if embedder._model is None
            else f"BAAI/bge-small-en-v1.5 with hash fallback, {embedder.dimension} dimensions"
        )
    report = build_report(
        results,
        dataset_kind=dataset_kind,
        dataset_label=dataset_label,
        document_count=len(docs),
        embedding_provider=embedding_label,
        generation_mode="Heuristic span extraction",
    )
    results_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_html_report(report), encoding="utf-8")
    print(render_console(report))
    print(f"\nReport     {report_path}")
    print(f"Results    {results_path}")
    print(f"Checkpoint {checkpoint_path}")
    print(f"Sample log {log_path}")


if __name__ == "__main__":
    main()
