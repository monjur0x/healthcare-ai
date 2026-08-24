#!/usr/bin/env python3
"""
Clinical Knowledge Ingestion Script

Ingests clinical guidelines, PubMed articles, WHO reports, CDC guidelines,
and hospital protocols into the RAG corpus.

Usage:
    python scripts/ingest_clinical_knowledge.py --config config/ingestion_config.json
    python scripts/ingest_clinical_knowledge.py --quick --topics diabetes heart
    python scripts/ingest_clinical_knowledge.py --quick --topics diabetes heart --sources kdigo acc_aha
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.corpus import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest clinical knowledge into RAG corpus")
    parser.add_argument("--config", type=str, help="Path to ingestion config JSON")
    parser.add_argument("--quick", action="store_true", help="Quick mode: ingest guidelines for specified topics")
    parser.add_argument("--topics", nargs="+", help="Topics to ingest (e.g., diabetes heart sepsis)")
    parser.add_argument("--sources", nargs="+", help="Sources to use (kdigo, acc_aha, nice, ers, pubmed, who, cdc)")
    parser.add_argument("--output", type=str, default="data/clinical_knowledge/ingested_documents.json", help="Output file path")
    parser.add_argument("--max-results", type=int, default=20, help="Max results per PubMed query")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    output_dir = Path("data/clinical_knowledge")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path("data/clinical_knowledge/ingested_documents.json")

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        from rag.data_ingestion import ClinicalKnowledgeIngestor
        ingestor = ClinicalKnowledgeIngestor(output_dir=Path(args.output))
        documents = ingestor.ingest_all_sources(config)
        logger.info(f"Ingested {len(documents)} documents from config")
    elif args.quick:
        if not args.topics:
            logger.error("--topics required with --quick")
            sys.exit(1)
        sources = args.sources or ["kdigo", "acc_aha", "nice", "who"]
        from rag.data_ingestion import quick_ingest_guidelines
        documents = quick_ingest_guidelines(topics=args.topics, sources=args.sources or ["kdigo", "acc_aha", "nice", "who"])
        logger.info(f"Quick ingested {len(documents)} documents for topics: {args.topics}")
    else:
        # Default: ingest core guidelines for major conditions
        topics = args.topics or ["diabetes", "heart", "kidney", "sepsis"]
        sources = args.sources or ["kdigo", "acc_aha", "nice", "who"]
        from rag.data_ingestion import quick_ingest_guidelines
        documents = quick_ingest_guidelines(topics, sources)
        logger.info(f"Quick ingested {len(documents)} documents for topics: {topics}")

    # Save documents
    output_dir = Path("data/clinical_knowledge")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path("data/clinical_knowledge/ingested_documents.json")

    docs_to_save = [
        Document(
            id=doc.id,
            text=doc.text,
            source=doc.source,
            metadata=doc.metadata
        )
        for doc in documents
    ]

    output_dir = Path("data/clinical_knowledge")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open("data/clinical_knowledge/ingested_documents.json", "w") as f:
        json.dump([{
            "id": doc.id,
            "text": doc.text,
            "source": doc.source,
            "metadata": doc.metadata
        } for doc in documents], f, indent=2)

    logger.info(f"Saved {len(documents)} documents to data/clinical_knowledge/ingested_documents.json")


if __name__ == "__main__":
    main()
