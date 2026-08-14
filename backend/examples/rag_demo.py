"""
End-to-end retrieval demo: text corpus -> chunking -> embeddings -> RAG.

Loads a directory of ``.txt``/``.md`` documents, splits them into
overlapping chunks, embeds the chunks (TF-IDF by default), indexes them
in an in-memory vector store, and answers queries by returning the most
relevant chunks with a prompt-ready context block. When a ground-truth
JSON file (query -> relevant document ids) is supplied, retrieval
quality metrics (precision@k, recall@k, MRR) are reported.

Usage (run from ``backend/``):

    python -m examples.rag_demo --corpus-dir path/to/docs \\
        --query "diabetes medication" --top-k 3

With ground truth for quality metrics:

    python -m examples.rag_demo --corpus-dir path/to/docs \\
        --ground-truth path/to/relevance.json --top-k 3

Each file in the corpus directory is one document; files under a
subdirectory use that subdirectory name as the source label.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from preprocessing.logger import get_logger
from rag import RAGPipeline, retrieval_metrics
from rag.documents import Document

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_corpus(corpus_dir: Path) -> list[Document]:
    """
    Load every text file under a directory as one document each.

    Parameters
    ----------
    corpus_dir : Path
        Directory (or file) containing the documents.

    Returns
    -------
    list[Document]
        Documents keyed by file name with the parent folder as source.
    """

    files = sorted(corpus_dir.rglob("*")) if corpus_dir.is_dir() else [corpus_dir]
    documents: list[Document] = []
    for path in files:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        documents.append(
            Document(
                id=path.name,
                text=path.read_text(encoding="utf-8"),
                source=path.parent.name if path.parent != corpus_dir else "",
            )
        )
    return documents


def load_ground_truth(path: Path | None) -> dict[str, list[str]]:
    """
    Load a query -> relevant document ids map.

    Parameters
    ----------
    path : Path | None
        Path to the JSON file, or None for no ground truth.

    Returns
    -------
    dict[str, list[str]]
        Query-to-relevant-ids mapping.
    """

    if path is None:
        return {}
    return {
        str(query): list(ids) for query, ids in json.loads(path.read_text()).items()
    }


def main(argv: list[str] | None = None) -> int:
    """Run the retrieval demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir", type=Path, required=True, help="Documents to ingest."
    )
    parser.add_argument(
        "--query", action="append", default=None, help="Query (repeatable)."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="JSON map of query -> relevant document ids for quality metrics.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/rag"))
    args = parser.parse_args(argv)

    documents = load_corpus(args.corpus_dir)
    if not documents:
        parser.error(f"No .txt/.md documents found under {args.corpus_dir}.")
    logger.info("Loaded %d documents", len(documents))

    pipeline = RAGPipeline(top_k=args.top_k)
    n_chunks = pipeline.ingest_documents(documents)
    logger.info("Indexed %d chunks", n_chunks)

    ground_truth = load_ground_truth(args.ground_truth)
    queries = args.query or list(ground_truth) or ["diabetes treatment"]
    if args.query is None and ground_truth:
        logger.info("Using ground-truth query set (%d queries)", len(queries))

    per_query: list[dict] = []
    for query in queries:
        results = pipeline.retrieve(query, top_k=args.top_k)
        entry = {
            "query": query,
            "results": [
                {"document_id": result.chunk.document_id, "score": result.score}
                for result in results
            ],
            "context": pipeline.build_context(query, top_k=args.top_k),
        }
        if query in ground_truth:
            metrics = retrieval_metrics(
                ground_truth[query],
                [result.chunk.document_id for result in results],
                k=args.top_k,
            )
            entry["metrics"] = metrics.to_dict()
        per_query.append(entry)

        logger.info("Query: %s", query)
        for result in results:
            logger.info("  [%.4f] %s", result.score, result.chunk.document_id)

    report = {
        "corpus_dir": str(args.corpus_dir),
        "n_documents": len(documents),
        "n_chunks": n_chunks,
        "top_k": args.top_k,
        "queries": per_query,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    logger.info("Artifacts written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
