"""Contract tests for the canonical n8n clinical workflow.

`n8n/clinical-full-v2.json` is the proposal §10 orchestration chain.
These tests pin its structure: every connection target exists, the
Agent 6 (risk-monitor) and Store Results steps are present, and every
backend URL the workflow calls maps to a route the FastAPI app actually
serves — so the workflow and the API cannot drift apart silently.
"""

from __future__ import annotations

import json
import re

from pathlib import Path

from api.config import APISettings
from api.main import create_app
from api.tests.test_api import FakeService

WORKFLOW = Path(__file__).resolve().parents[3] / "n8n" / "clinical-full-v2.json"

BACKEND_URL_RE = re.compile(r"/api/v1/([A-Za-z0-9_{}/-]+)")


def _workflow() -> dict:
    return json.loads(WORKFLOW.read_text())


def _route_paths() -> set[str]:
    # Read the OpenAPI schema: current FastAPI keeps included-router
    # routes inside an _IncludedRouter entry, so app.routes alone
    # under-reports the served paths.
    app = create_app(cfg=APISettings(_env_file=None), service=FakeService())
    return set(app.openapi()["paths"])


def test_workflow_connections_resolve():
    workflow = _workflow()
    names = [node["name"] for node in workflow["nodes"]]
    ids = [node["id"] for node in workflow["nodes"]]
    assert len(ids) == len(set(ids)), "duplicate node ids"
    for source, outputs in workflow["connections"].items():
        assert source in names, f"connection source {source!r} is not a node"
        for branch in outputs["main"]:
            for target in branch:
                assert target["node"] in names, (
                    f"{source!r} points at missing node {target['node']!r}"
                )


def test_workflow_contains_risk_monitor_and_store_steps():
    workflow = _workflow()
    by_name = {node["name"]: node for node in workflow["nodes"]}
    assert "8.Risk Monitor" in by_name
    assert "Store Report" in by_name

    chain = workflow["connections"]
    assert chain["7.Explainability Expert"]["main"][0][0]["node"] == ("8.Risk Monitor")
    assert chain["8.Risk Monitor"]["main"][0][0]["node"] == "Assemble Report"
    assert chain["Assemble Report"]["main"][0][0]["node"] == "Store Report"
    assert chain["Store Report"]["main"][0][0]["node"] == "Attach Store ID"
    assert chain["Attach Store ID"]["main"][0][0]["node"] == "IF:High Risk?"


def test_workflow_backend_urls_match_served_routes():
    workflow = _workflow()
    routes = _route_paths()
    for node in workflow["nodes"]:
        url = (node.get("parameters") or {}).get("url", "")
        if "/api/v1/" not in url:
            continue  # external webhook (doctor notify) or non-HTTP node
        match = BACKEND_URL_RE.search(url)
        assert match, f"unparseable backend URL on {node['name']!r}"
        path = "/api/v1/" + match.group(1).rstrip("}'\"")
        assert path in routes, (
            f"{node['name']!r} calls {path} which the API does not serve"
        )


def _node(workflow: dict, name: str) -> dict:
    return next(node for node in workflow["nodes"] if node["name"] == name)


def test_rag_query_builder_is_disease_aware():
    workflow = _workflow()
    js = _node(workflow, "Build RAG Query")["parameters"]["jsCode"]
    # No hardcoded condition: the query anchors on the predictor's
    # disease output, mirroring backend build_disease_query.
    assert "diabetes treatment" not in js
    assert "pred.disease" in js
    assert "clinical guidelines diagnosis management treatment" in js
    assert "prevention risk factors screening guidelines" in js


def test_evidence_retrieval_receives_built_query():
    workflow = _workflow()
    body = _node(workflow, "5.Medical Researcher")["parameters"]["jsonBody"]
    assert "$json.query" in body, (
        "Medical Researcher must forward the query-builder output"
    )


def test_disease_predictor_feeds_query_builder():
    workflow = _workflow()
    chain = workflow["connections"]
    assert chain["4.Disease Predictor"]["main"][0][0]["node"] == "Build RAG Query"
    assert chain["Build RAG Query"]["main"][0][0]["node"] == ("5.Medical Researcher")
