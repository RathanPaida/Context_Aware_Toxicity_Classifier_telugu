"""Bounded localhost HTTP smoke check against the actual saved CPU demo model."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contextual.config import write_json


def main():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = {**os.environ, 'TOXICITY_CONFIG': 'configs/demo.json'}
    log = ROOT / 'scratch/submission_http.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.server:app', '--host',
            '127.0.0.1', '--port', str(port), '--log-level', 'warning'], cwd=ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            base = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 60
            health = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited; see {log}')
                try:
                    response = requests.get(base+'/api/health', timeout=2)
                    if response.status_code == 200:
                        health = response.json()
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            assert health and health['ready'], health or f'No server response; see {log}'
            assert requests.get(base+'/', timeout=5).status_code == 200
            assert requests.get(base+'/docs', timeout=5).status_code == 200
            result = requests.post(base+'/api/moderate', json={'text': 'Thank you for explaining.',
                'history': ['Let me explain.']}, timeout=30)
            assert result.status_code == 200, result.text
            assert 0 <= result.json()['toxicity_score'] <= 1
            receipt = {'method': 'real uvicorn localhost HTTP', 'health': health,
                       'index_status': 200, 'docs_status': 200,
                       'prediction_status': result.status_code,
                       'toxicity_score': result.json()['toxicity_score']}
            write_json(ROOT / 'reports/submission_http_verification.json', receipt)
            print(json.dumps(receipt, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == '__main__':
    main()
