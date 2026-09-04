#!/usr/bin/env python3
"""
Demo startup script - launches all components for end-to-end demonstration.

Starts:
1. FastAPI backend (port 8000)
2. n8n orchestration (port 5678)
3. Streamlit dashboard (port 8501)
4. Doctor notification receiver (port 9999)

Usage:
    python scripts/start_demo.py
    python scripts/start_demo.py --stop   # stop all demo processes
    python scripts/start_demo.py --status # show running status
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_DIR = REPO_ROOT / "frontend"
N8N_DIR = REPO_ROOT / "n8n"

# Process tracking
PROCESSES = {}

# Configuration
CONFIG = {
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "log": "/tmp/demo_api.log",
    },
    "n8n": {
        "host": "127.0.0.1",
        "port": 5678,
        "log": "/tmp/demo_n8n.log",
    },
    "dashboard": {
        "host": "127.0.0.1",
        "port": 8501,
        "log": "/tmp/demo_dashboard.log",
    },
    "notify_receiver": {
        "host": "127.0.0.1",
        "port": 9999,
        "log": "/tmp/demo_notify.log",
    },
}


def get_venv_python():
    """Get the venv python path."""
    venv = BACKEND_DIR / "CrewAI" / ".venv-opencode"
    if venv.exists():
        return venv / "bin" / "python"
    # Fallback
    return sys.executable


def get_n8n_bin():
    """Get n8n binary path."""
    # Try common locations
    for path in [
        Path.home() / ".config" / "nvm" / "versions" / "node" / "v24.16.0" / "bin" / "n8n",
        Path("/usr/local/bin/n8n"),
        Path("/usr/bin/n8n"),
    ]:
        if path.exists():
            return path
    # Try from PATH
    result = subprocess.run(["which", "n8n"], capture_output=True, text=True)
    if result.returncode == 0:
        return Path(result.stdout.strip())
    return None


def build_env():
    """Build environment variables for demo."""
    env = os.environ.copy()
    env.update({
        # API settings
        "CREW_LLM_API_KEY": "",
        "CREW_LLM_BASE_URL": "",
        "API_MODEL_PATH": str(REPO_ROOT / "backend" / "artifacts" / "multi_disease" / "global_model.joblib"),
        "DATASET_DIR": str(REPO_ROOT / "dataset"),
        "N8N_BLOCK_ENV_ACCESS_IN_NODE": "false",
        "DOCTOR_NOTIFY_WEBHOOK": f"http://{CONFIG['notify_receiver']['host']}:{CONFIG['notify_receiver']['port']}/doctor-notify",
    })
    return env


def start_notify_receiver():
    """Start the doctor notification receiver HTTP server."""
    python = get_venv_python()
    script = REPO_ROOT / "scripts" / "notify_receiver.py"
    
    # Create receiver script if it doesn't exist
    if not script.exists():
        script.write_text('''#!/usr/bin/env python3
"""Simple HTTP server to receive doctor notifications."""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys

class NotifyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        print(f"NOTIFICATION RECEIVED:", file=sys.stderr)
        print(json.dumps(json.loads(body), indent=2), file=sys.stderr)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"received": true}')
    
    def log_message(self, format, *args):
        pass  # Suppress default logging

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 9999), NotifyHandler)
    print("Doctor notification receiver started on http://127.0.0.1:9999", file=sys.stderr)
    server.serve_forever()
''')
    
    proc = subprocess.Popen(
        [str(python), str(script)],
        stdout=open(CONFIG["notify_receiver"]["log"], "a"),
        stderr=subprocess.STDOUT,
        env=build_env(),
    )
    PROCESSES["notify_receiver"] = proc
    print(f"✓ Notification receiver started (PID: {proc.pid}) on port {CONFIG['notify_receiver']['port']}")
    time.sleep(1)


def start_api():
    """Start FastAPI backend."""
    python = get_venv_python()
    env = build_env()
    
    proc = subprocess.Popen(
        [str(python), "-m", "uvicorn", "api.main:app",
         "--host", CONFIG["api"]["host"],
         "--port", str(CONFIG["api"]["port"])],
        cwd=BACKEND_DIR,
        stdout=open(CONFIG["api"]["log"], "a"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    PROCESSES["api"] = proc
    print(f"✓ FastAPI started (PID: {proc.pid}) on http://{CONFIG['api']['host']}:{CONFIG['api']['port']}")


def start_n8n():
    """Start n8n orchestration."""
    n8n_bin = get_n8n_bin()
    if not n8n_bin:
        print("⚠ n8n not found. Install with: npm install -g n8n")
        return
    
    env = build_env()
    env["N8N_USER_FOLDER"] = "/tmp/n8n-demo-data"
    
    proc = subprocess.Popen(
        [str(n8n_bin), "start", "--port", str(CONFIG["n8n"]["port"])],
        stdout=open(CONFIG["n8n"]["log"], "a"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    PROCESSES["n8n"] = proc
    print(f"✓ n8n started (PID: {proc.pid}) on http://{CONFIG['n8n']['host']}:{CONFIG['n8n']['port']}")


def start_dashboard():
    """Start Streamlit dashboard."""
    python = get_venv_python()
    dashboard_script = FRONTEND_DIR / "streamlit_app.py"
    
    if not dashboard_script.exists():
        print(f"⚠ Dashboard not found at {dashboard_script}")
        return
    
    env = build_env()
    proc = subprocess.Popen(
        [str(python), "-m", "streamlit", "run", str(dashboard_script),
         "--server.address", CONFIG["dashboard"]["host"],
         "--server.port", str(CONFIG["dashboard"]["port"]),
         "--server.headless", "true"],
        cwd=FRONTEND_DIR,
        stdout=open(CONFIG["dashboard"]["log"], "a"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    PROCESSES["dashboard"] = proc
    print(f"✓ Streamlit started (PID: {proc.pid}) on http://{CONFIG['dashboard']['host']}:{CONFIG['dashboard']['port']}")


def wait_for_services():
    """Wait for all services to be healthy."""
    import urllib.request
    
    services = [
        ("FastAPI", f"http://{CONFIG['api']['host']}:{CONFIG['api']['port']}/health"),
        ("n8n", f"http://{CONFIG['n8n']['host']}:{CONFIG['n8n']['port']}/healthz"),
        ("Dashboard", f"http://{CONFIG['dashboard']['host']}:{CONFIG['dashboard']['port']}/_stcore/health"),
    ]
    
    print("\nWaiting for services to be ready...")
    for name, url in services:
        for i in range(30):
            try:
                req = urllib.request.Request(url)
                urllib.request.urlopen(req, timeout=2)
                print(f"  ✓ {name} ready")
                break
            except Exception:
                if i == 29:
                    print(f"  ⚠ {name} not ready after 30s")
                time.sleep(1)


def import_n8n_workflows():
    """Import n8n workflows."""
    n8n_bin = get_n8n_bin()
    if not n8n_bin:
        return
    
    env = build_env()
    env["N8N_USER_FOLDER"] = "/tmp/n8n-demo-data"
    
    workflows = [
        ("Healthcare End-to-End", N8N_DIR / "healthcare-endtoend.json"),
        ("Clinical Full Pipeline (per-agent)", N8N_DIR / "clinical-full-v2.json"),
        ("Feedback Retrain", N8N_DIR / "feedback-retrain.json"),
        ("Risk Monitoring", N8N_DIR / "risk-monitoring.json"),
    ]
    
    print("\nImporting n8n workflows...")
    for name, path in workflows:
        if not path.exists():
            print(f"  ⚠ {name}: not found at {path}")
            continue
        # Import workflow (wraps in array with ID for n8n 2.x)
        import json
        import uuid
        wf = json.loads(path.read_text())
        wf["id"] = str(uuid.uuid4())
        wf["active"] = False
        import_file = f"/tmp/{path.stem}_import.json"
        Path(import_file).write_text(json.dumps([wf], indent=2))
        
        subprocess.run(
            [str(n8n_bin), "import:workflow", "--input", import_file],
            env=env,
            capture_output=True,
        )
        print(f"  ✓ {name} imported")
    
    # Activate workflows via DB
    activate_workflows()


def activate_workflows():
    """Activate imported workflows in n8n DB."""
    import sqlite3
    import uuid
    
    db_path = Path("/tmp/n8n-demo-data/.n8n/database.sqlite")
    if not db_path.exists():
        return
    
    conn = sqlite3.connect(str(db_path))
    
    # Find and activate each workflow
    for name in [
        "Healthcare End-to-End Pipeline",
        "Clinical Full Pipeline v2",
        "Feedback-Driven Retrain",
        "Risk Monitoring",
    ]:
        row = conn.execute(
            "SELECT id FROM workflow_entity WHERE name = ?", (name,)
        ).fetchone()
        if row:
            wf_id = row[0]
            version_id = str(uuid.uuid4())
            # Create workflow_history entry
            conn.execute("""
                INSERT INTO workflow_history (versionId, workflowId, authors, createdAt, updatedAt, nodes, connections, name, autosaved, description, nodeGroups)
                SELECT ?, id, '[]', createdAt, updatedAt, nodes, connections, name, 'false', description, '[]'
                FROM workflow_entity WHERE id = ?
            """, (version_id, wf_id))
            conn.execute("UPDATE workflow_entity SET active = 1, activeVersionId = ? WHERE id = ?", (version_id, wf_id))
            print(f"  ✓ {name} activated")
    
    conn.commit()
    conn.close()


def print_demo_info():
    """Print demo access information."""
    print("\n" + "="*60)
    print("🏥 HEALTHCARE AI DEMO - ALL SERVICES RUNNING")
    print("="*60)
    print(f"\n📊 FastAPI Backend:     http://{CONFIG['api']['host']}:{CONFIG['api']['port']}")
    print(f"   - Health:            http://{CONFIG['api']['host']}:{CONFIG['api']['port']}/health")
    print(f"   - API Docs:          http://{CONFIG['api']['host']}:{CONFIG['api']['port']}/docs")
    print(f"   - Analyze endpoint:  POST http://{CONFIG['api']['host']}:{CONFIG['api']['port']}/api/v1/analyze")
    print(f"   - Risk history:      GET  http://{CONFIG['api']['host']}:{CONFIG['api']['port']}/api/v1/risk/history")
    print(f"   - Feedback:          POST http://{CONFIG['api']['host']}:{CONFIG['api']['port']}/api/v1/feedback")
    
    print(f"\n🔄 n8n Orchestration:   http://{CONFIG['n8n']['host']}:{CONFIG['n8n']['port']}")
    print(f"   - Editor UI:         http://{CONFIG['n8n']['host']}:{CONFIG['n8n']['port']}")
    print(f"   - End-to-end webhook: POST http://{CONFIG['n8n']['host']}:{CONFIG['n8n']['port']}/webhook/healthcare-endtoend")
    print(f"   - Feedback webhook:   POST http://{CONFIG['n8n']['host']}:{CONFIG['n8n']['port']}/webhook/feedback-retrain")
    
    print(f"\n📈 Streamlit Dashboard:  http://{CONFIG['dashboard']['host']}:{CONFIG['dashboard']['port']}")
    
    print(f"\n🔔 Doctor Notify Receiver: http://{CONFIG['notify_receiver']['host']}:{CONFIG['notify_receiver']['port']}")
    print(f"   (check logs: tail -f {CONFIG['notify_receiver']['log']})")
    
    print("\n" + "="*60)
    print("QUICK DEMO COMMANDS:")
    print("="*60)
    print("""
# 1. Full end-to-end analysis via n8n (triggers train + analyze + notify if high risk)
curl -X POST http://127.0.0.1:5678/webhook/healthcare-endtoend \\
  -H "Content-Type: application/json" \\
  -d '{"preset": "multi_disease", "patient": {"name": "Demo Patient", "study_id": "DEMO001", "age": 55},
       "features": {"pregnancies": 6, "glucose": 190, "blood_pressure": 92, "skin_thickness": 35,
                    "insulin": 180, "bmi": 42, "multi_disease_pedigree_function": 1.2, "age": 55}}'

# 2. Direct API analyze (low risk - no notification)
curl -X POST http://127.0.0.1:8000/api/v1/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"preset": "multi_disease", "patient": {"name": "Healthy", "study_id": "HLT001", "age": 30},
       "features": {"pregnancies": 1, "glucose": 85, "blood_pressure": 70, "skin_thickness": 20,
                    "insulin": 50, "bmi": 22, "multi_disease_pedigree_function": 0.2, "age": 30}}'

# 3. Submit clinician feedback
curl -X POST http://127.0.0.1:8000/api/v1/feedback \\
  -H "Content-Type: application/json" \\
  -d '{"preset": "multi_disease", "patient_id": "FB001",
       "features": {"pregnancies": 6, "glucose": 190, "blood_pressure": 92, "skin_thickness": 35,
                    "insulin": 180, "bmi": 42, "multi_disease_pedigree_function": 1.2, "age": 55},
       "confirmed_label": 1, "predicted_label": 1, "confidence": 0.95}'

# 4. Check risk history & alerts
curl http://127.0.0.1:8000/api/v1/risk/history
curl http://127.0.0.1:8000/api/v1/risk/alerts

# 5. Trigger feedback retrain via n8n
curl -X POST http://127.0.0.1:5678/webhook/feedback-retrain \\
  -H "Content-Type: application/json" \\
  -d '{"preset": "multi_disease"}'
""")
    print("\nPress Ctrl+C to stop all services\n")


def stop_all():
    """Stop all demo processes."""
    print("\nStopping all demo services...")
    for name, proc in PROCESSES.items():
        if proc.poll() is None:
            print(f"  Stopping {name} (PID: {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print(f"  ✓ {name} stopped")


def show_status():
    """Show status of demo processes."""
    print("Demo Service Status:")
    for name, proc in PROCESSES.items():
        status = "RUNNING" if proc.poll() is None else f"STOPPED (exit: {proc.returncode})"
        print(f"  {name:20s} PID: {proc.pid:6d}  {status}")
    
    # Check if services are responding
    import urllib.request
    services = [
        ("FastAPI", f"http://{CONFIG['api']['host']}:{CONFIG['api']['port']}/health"),
        ("n8n", f"http://{CONFIG['n8n']['host']}:{CONFIG['n8n']['port']}/healthz"),
        ("Dashboard", f"http://{CONFIG['dashboard']['host']}:{CONFIG['dashboard']['port']}/_stcore/health"),
    ]
    for name, url in services:
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"  {name:20s} HTTP: HEALTHY")
        except Exception:
            print(f"  {name:20s} HTTP: UNREACHABLE")


def main():
    parser = argparse.ArgumentParser(description="Healthcare AI Demo Launcher")
    parser.add_argument("--stop", action="store_true", help="Stop all demo services")
    parser.add_argument("--status", action="store_true", help="Show service status")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for services to be ready")
    parser.add_argument("--no-workflows", action="store_true", help="Skip n8n workflow import")
    args = parser.parse_args()
    
    if args.stop:
        stop_all()
        return
    
    if args.status:
        show_status()
        return
    
    print("🚀 Starting Healthcare AI Demo...")
    print(f"Repo: {REPO_ROOT}")
    
    # Start all services
    start_notify_receiver()
    start_api()
    start_n8n()
    start_dashboard()
    
    if not args.no_wait:
        wait_for_services()
    
    if not args.no_workflows:
        import_n8n_workflows()
    
    print_demo_info()
    
    # Keep running until Ctrl+C
    try:
        while True:
            time.sleep(1)
            # Check if any process died
            for name, proc in PROCESSES.items():
                if proc.poll() is not None:
                    print(f"\n⚠ {name} exited unexpectedly (code: {proc.returncode})")
    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()