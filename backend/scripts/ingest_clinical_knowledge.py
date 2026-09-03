#!/usr/bin/env python3
"""
Clinical Knowledge Ingestion Script

Ingests clinical guidelines, PubMed articles, WHO reports, CDC guidelines,
and hospital protocols into the RAG corpus.

Usage:
    python scripts/ingest_clinical_knowledge.py --config config/ingestion_config.json
    python scripts/ingest_clinical_knowledge.py --quick --topics diabetes heart
    python scripts/ingest_clinical_knowledge.py --quick --topics diabetes heart \
        --sources kdigo acc_aha
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "data/clinical_knowledge/ingested_documents.json"
DEFAULT_SOURCES = ["kdigo", "acc_aha", "nice", "who"]


def main() -> None:
    """Parse CLI arguments and ingest clinical knowledge sources."""
    parser = argparse.ArgumentParser(
        description="Ingest clinical knowledge into RAG corpus"
    )
    parser.add_argument("--config", type=str, help="Path to ingestion config JSON")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: ingest guidelines for specified topics",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Topics to ingest (e.g., diabetes heart sepsis)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Sources to use (kdigo, acc_aha, nice, ers, pubmed, who, cdc)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output file path",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Max results per PubMed query",
    )

    args = parser.parse_args()
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.config:
        with open(args.config) as handle:
            config = json.load(handle)
        from rag.data_ingestion import ClinicalKnowledgeIngestor

        ingestor = ClinicalKnowledgeIngestor(output_dir=output_file.parent)
        documents = ingestor.ingest_all_sources(config)
        logger.info("Ingested %d documents from config", len(documents))
    elif args.quick:
        if not args.topics:
            logger.error("--topics required with --quick")
            sys.exit(1)
        from rag.data_ingestion import quick_ingest_guidelines

        sources = args.sources or DEFAULT_SOURCES
        documents = quick_ingest_guidelines(topics=args.topics, sources=sources)
        logger.info(
            "Quick ingested %d documents for topics: %s", len(documents), args.topics
        )
    else:
        # Default: ingest core guidelines for major conditions
        from rag.data_ingestion import quick_ingest_guidelines

        topics = args.topics or ["diabetes", "heart", "kidney", "sepsis"]
        sources = args.sources or DEFAULT_SOURCES
        documents = quick_ingest_guidelines(topics, sources)
        logger.info(
            "Quick ingested %d documents for topics: %s", len(documents), topics
        )

    with open(output_file, "w") as handle:
        json.dump(
            [
                {
                    "id": doc.id,
                    "text": doc.text,
                    "source": doc.source,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ],
            handle,
            indent=2,
        )
    logger.info("Saved %d documents to %s", len(documents), output_file)


if __name__ == "__main__":
    main()
