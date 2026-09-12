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
import shutil
import subprocess

from pathlib import Path

import pytest

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


ENDTOEND = Path(__file__).resolve().parents[3] / "n8n" / "healthcare-endtoend.json"


def _endtoend() -> dict:
    return json.loads(ENDTOEND.read_text())


def test_branch_after_train_uses_webhook_scope():
    """The CSV-branch decision must read the webhook payload, not train output.

    Regression guard: "IF: CSV Input?" sits downstream of "HTTP: Train
    Model". Its condition used to be ``!!$json.body.csv_b64`` — but after a
    training run ``$json`` is the *train response* (no ``body.csv_b64``),
    so every CSV upload that triggered a retrain silently fell through to
    the plain-patient branch with ``features: {}`` and died with
    ``422 Prediction requires feature row``. The condition must name the
    webhook trigger explicitly so it is scope-proof.
    """
    workflow = _endtoend()
    chain = workflow["connections"]
    assert chain["HTTP: Train Model"]["main"][0][0]["node"] == "IF: CSV Input?"
    conditions = _node(workflow, "IF: CSV Input?")["parameters"]["conditions"][
        "conditions"
    ]
    assert conditions, "IF: CSV Input? has no conditions"
    for cond in conditions:
        left = cond.get("leftValue", "")
        assert "$(" in left and "Webhook" in left, (
            f"branch condition {left!r} reads bare $json — it breaks when "
            "an HTTP node feeds the IF node"
        )
        assert "$json.body" not in left, (
            f"branch condition {left!r} still depends on input scope"
        )


def test_no_if_node_trusts_http_input_scope():
    """Generalize the guard: no IF node fed by an HTTP node may branch on
    bare ``$json`` — its input item changes with whatever ran upstream."""
    for path in (WORKFLOW, ENDTOEND):
        workflow = json.loads(path.read_text())
        http_names = {
            n["name"]
            for n in workflow["nodes"]
            if n["type"] == "n8n-nodes-base.httpRequest"
        }
        preds: dict[str, list[str]] = {}
        for src, outputs in workflow["connections"].items():
            for branch in outputs.get("main", []):
                for target in branch:
                    preds.setdefault(target["node"], []).append(src)
        for node in workflow["nodes"]:
            if node["type"] != "n8n-nodes-base.if":
                continue
            if not any(p in http_names for p in preds.get(node["name"], [])):
                continue  # IF fed only by webhook/code — bare $json is fine
            for cond in (
                (node.get("parameters") or {})
                .get("conditions", {})
                .get("conditions", [])
            ):
                left = cond.get("leftValue", "")
                if "$json" in left:
                    assert "$(" in left, (
                        f"{path.name}: IF node {node['name']!r} branches on "
                        f"bare $json ({left!r}) but is fed by an HTTP node"
                    )


def test_code_nodes_are_valid_javascript(tmp_path):
    """Every Code node's jsCode must parse under node.

    Regression guard: a one-character typo in "Assemble Report"
    (``json||{}}]``) shipped unnoticed and failed every v2 execution at
    runtime with ``SyntaxError: Unexpected token ']'``. String
    assertions cannot catch that class of bug — parsing can.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node binary not available")
    workflow = _workflow()
    code_nodes = [
        (n["name"], (n.get("parameters") or {}).get("jsCode"))
        for n in workflow["nodes"]
        if (n.get("parameters") or {}).get("jsCode")
    ]
    assert code_nodes, "expected Code nodes in the canonical workflow"
    for name, js in code_nodes:
        probe = tmp_path / f"{name}.js".replace(" ", "_").replace(":", "")
        # Code nodes run with $json/$() in scope and allow a top-level
        # `return`; embed the raw source in a function so node parses the
        # payload itself (wrapping it in a string literal would only check
        # the wrapper and miss payload typos entirely).
        probe.write_text("async function __probe($json, $) {\n" + js + "\n}")
        result = subprocess.run(
            [node, "--check", str(probe)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{name!r} jsCode does not parse: {result.stderr.strip()}"
        )


def test_assemble_report_forwards_prediction_enrichment():
    """Assemble Report must carry the predictor's disease/label enrichment.

    Regression guard: the node rebuilt ``prediction`` from the Disease
    Predictor output but dropped ``disease`` / ``predicted_label``, so
    every stored v2 report identified its prediction only by raw class
    (``"1"``/``"0"``) while the end-to-end report carried the readable
    enrichment (``"sepsis"`` / ``"No Sepsis"``).
    """
    workflow = _workflow()
    js = _node(workflow, "Assemble Report")["parameters"]["jsCode"]
    assert "disease:pred.disease" in js.replace(" ", ""), (
        "Assemble Report drops the predictor's disease enrichment"
    )
    assert "predicted_label:pred.predicted_label" in js.replace(" ", ""), (
        "Assemble Report drops the predictor's readable label"
    )
