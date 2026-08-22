"""Data Ingestion Pipeline for Clinical Knowledge Sources."""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any, ClassVar

import requests

from bs4 import BeautifulSoup

from rag.corpus import Document

logger = logging.getLogger(__name__)


def quick_ingest_guidelines(
    topics: list[str], sources: list[str] | None = None
) -> list[Document]:
    """Quick ingestion of guidelines for specified topics from given sources."""
    if sources is None:
        sources = ["kdigo", "acc_aha", "nice", "who"]

    ingestor = ClinicalGuidelineIngestor()
    documents = []

    topic_map = {
        "diabetes": [("kdigo", "diabetes"), ("who", "diabetes")],
        "heart": [
            ("acc_aha", "hf"),
            ("acc_aha", "hypertension"),
            ("nice", "heart_failure"),
        ],
        "kidney": [("kdigo", "ckd"), ("kdigo", "aki"), ("nice", "ckd"), ("who", "ckd")],
        "sepsis": [
            ("surviving_sepsis", "sepsis"),
            ("who", "sepsis"),
            ("nice", "sepsis"),
        ],
        "aki": [("kdigo", "aki"), ("nice", "aki")],
        "ckd": [("kdigo", "ckd"), ("nice", "ckd"), ("who", "ckd")],
    }

    available_sources = set(sources) if sources else None

    for topic in topics:
        if topic not in topic_map:
            logger.warning(f"No guideline mapping for topic: {topic}")
            continue
        for source, topic_key in topic_map[topic]:
            if available_sources and source not in available_sources:
                continue
            html = ingestor.fetch_guideline(source, topic_key)
            if not html:
                continue
            content = ingestor.extract_text(html, source)
            if not content or len(content) < 200:
                continue
            documents.append(
                Document(
                    id=f"guideline_{source}_{topic}",
                    text=content,
                    source=f"guideline:{source}",
                )
            )
    return documents


def ingest_clinical_knowledge(config: dict[str, Any]) -> list[Document]:
    """High-level function to ingest clinical knowledge from config."""
    from rag.data_ingestion import ClinicalKnowledgeIngestor

    ingestor = ClinicalKnowledgeIngestor()
    return ingestor.ingest_all_sources(config)


class ClinicalGuidelineIngestor:
    """Fetches and parses clinical practice guidelines."""

    GUIDELINE_URLS: ClassVar[dict[str, dict[str, str]]] = {
        "kdigo": {
            "ckd": "https://kdigo.org/guidelines/ckd/",
            "aki": "https://kdigo.org/guidelines/aki/",
            "anemia": "https://kdigo.org/guidelines/anemia/",
            "diabetes": "https://kdigo.org/guidelines/diabetes/",
        },
        "nice": {
            "ckd": "https://www.nice.org.uk/guidance/ng203",
            "diabetes": "https://www.nice.org.uk/guidance/ng28",
            "heart_failure": "https://www.nice.org.uk/guidance/ng106",
            "sepsis": "https://www.nice.org.uk/guidance/ng51",
        },
        "who": {
            "sepsis": "https://www.who.int/publications/i/item/9789240019678",
            "diabetes": "https://www.who.int/publications/i/item/9789240007058",
            "ckd": "https://www.who.int/publications/i/item/9789240007126",
        },
    }

    def __init__(self, cache_dir: str | Path = "data/guidelines_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_guideline(self, source: str, topic: str) -> str | None:
        urls = self.GUIDELINE_URLS.get(source, {})
        url = urls.get(topic)
        if not url:
            return None
        cache_file = self.cache_dir / f"{source}_{topic}.html"
        if cache_file.exists():
            return cache_file.read_text()
        try:
            resp = requests.get(
                url, headers={"User-Agent": "Healthcare-AI/1.0"}, timeout=30
            )
            resp.raise_for_status()
            cache_file.write_text(resp.text)
            return resp.text
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to fetch {source}/{topic}: {e}")
            return None

    def extract_text(self, html: str, source: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for el in soup(["script", "style", "nav", "footer"]):
            el.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        return "\n".join(ln for ln in lines if len(ln) > 20)


class ClinicalKnowledgeIngestor:
    """Orchestrates ingestion from all clinical knowledge sources."""

    def __init__(self, output_dir: str | Path = "data/clinical_knowledge"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.guideline_ingestor = ClinicalGuidelineIngestor()
        self.pubmed_ingestor = PubMedIngestor()

    def ingest_all_sources(self, config: dict[str, Any]) -> list[Document]:
        docs: list[Document] = []
        if "guidelines" in config:
            for source, topics in config["guidelines"].items():
                for topic in topics:
                    html = self.guideline_ingestor.fetch_guideline(source, topic)
                    if not html:
                        continue
                    content = self.guideline_ingestor.extract_text(html, source)
                    if content and len(content) > 200:
                        docs.append(
                            Document(
                                id=f"guideline_{source}_{topic}",
                                text=content,
                                source=f"guideline:{source}",
                            )
                        )
        if "pubmed_queries" in config:
            for query in config["pubmed_queries"]:
                articles = self.pubmed_ingestor.search_and_fetch(query, max_results=10)
                for a in articles:
                    docs.append(
                        Document(
                            id=f"pubmed_{a['pmid']}",
                            text=f"{a['title']}\n\n{a['abstract']}",
                            source="pubmed",
                        )
                    )
        return docs


class PubMedIngestor:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, query: str, max_results: int = 20) -> list[str]:
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }
        resp = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])

    def fetch_abstracts(self, pmids: list[str]) -> list[dict[str, Any]]:
        if not pmids:
            return []
        params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        resp = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params, timeout=30)
        soup = BeautifulSoup(resp.content, "xml")
        results = []
        for article in soup.find_all("PubmedArticle"):
            pmid_el = article.find("PMID")
            title_el = article.find("ArticleTitle")
            abstract_el = article.find("AbstractText")
            if pmid_el and abstract_el:
                results.append(
                    {
                        "pmid": pmid_el.text,
                        "title": title_el.text if title_el else "",
                        "abstract": abstract_el.get_text(strip=True),
                    }
                )
        return results

    def search_and_fetch(
        self, query: str, max_results: int = 20
    ) -> list[dict[str, Any]]:
        pmids = self.search(query, max_results=min(max_results, 50))
        return self.fetch_abstracts(pmids)
