#!/usr/bin/env python3
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
