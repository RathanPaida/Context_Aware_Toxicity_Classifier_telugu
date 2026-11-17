"""Run review notebook cells and real checkpoint API checks.

This executes genuine notebook code in order in Python, records outputs and cell
times, and explicitly labels the method. --live-kernel selects real Jupyter
execution; neither mode claims a visual browser test.
"""
import ast
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['TOXICITY_PROJECT_ROOT'] = str(ROOT)
os.environ['TOXICITY_RUN_TRAINING'] = '0'
os.environ['TOXICITY_RUN_LORA'] = '0'
os.environ.pop('TOXICITY_CONFIG', None)


def run_notebook(name, live_kernel=False):
    path = ROOT / name
    notebook = nbformat.read(path, as_version=4)
    if live_kernel:
        # Use a workspace-local profile on Windows rather than changing permissions
        # on the user's normal IPython profile.
        os.environ['IPYTHONDIR'] = str(ROOT / 'scratch/verification_ipython')
        if os.name == 'nt':
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        from nbclient import NotebookClient
        from datetime import datetime
        notebook = NotebookClient(notebook, timeout=600, kernel_name='python3',
                                  resources={'metadata': {'path': str(ROOT)}}).execute()
        timings = []
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code': continue
            execution = cell.metadata.get('execution', {})
            started = execution.get('iopub.status.busy')
            ended = execution.get('iopub.status.idle')
            seconds = (datetime.fromisoformat(ended.replace('Z', '+00:00')) -
                       datetime.fromisoformat(started.replace('Z', '+00:00'))).total_seconds() if started and ended else None
            timings.append({'cell': index+1, 'seconds': seconds})
        notebook.metadata['verification'] = {'method': 'live Jupyter kernel via nbclient', 'training': False}
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f'{name}: live kernel passed', flush=True)
        return timings
    namespace = {'__name__': '__main__'}
    count = 0
    timings = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        count += 1
        started = time.perf_counter()
        output = io.StringIO()
        tree = ast.parse(cell.source)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(compile(tree, f'{name}:cell{index+1}', 'exec'), namespace)
            result = eval(compile(ast.Expression(last.value), name, 'eval'), namespace) if last else None
        cell.execution_count = count
        cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=output.getvalue())] if output.getvalue() else []
        if result is not None:
            data = {'text/plain': repr(result)}
            if hasattr(result, '_repr_html_'):
                html = result._repr_html_()
                if html: data['text/html'] = html
            cell.outputs.append(nbformat.v4.new_output('execute_result', data=data, execution_count=count))
        elapsed = time.perf_counter() - started
        cell.metadata['verification_seconds'] = round(elapsed, 3)
        timings.append({'cell': index+1, 'seconds': round(elapsed, 3)})
        print(f'{name} cell {index+1}: {elapsed:.2f}s', flush=True)
    notebook.metadata['verification'] = {'method': 'sequential in-process Python; not live Jupyter kernel', 'training': False}
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-kernel', action='store_true', help='Execute with a real Jupyter kernel instead of in-process Python')
    args = parser.parse_args()
    timings = {name: run_notebook(name, args.live_kernel) for name in
               ('Context_Aware_Toxicity.ipynb', 'LoRA_Fine_Tuning.ipynb', 'Data_Downloads.ipynb')}
    from fastapi.testclient import TestClient
    from app.server import app
    with TestClient(app) as client:
        health = client.get('/api/health')
        assert health.status_code == 200 and health.json()['ready'], health.text
        single = client.post('/api/moderate', json={'text': 'Thank you for explaining.', 'history': ['Let me explain.']})
        assert single.status_code == 200, single.text
        assert 0 <= single.json()['toxicity_score'] <= 1
        messages = [{'id': 'p', 'parent_id': None, 'conversation_id': 'demo', 'text': 'Let me explain.'},
                    {'id': 't', 'parent_id': 'p', 'conversation_id': 'demo', 'text': 'Thank you for explaining.'}]
        thread = client.post('/api/thread', json={'messages': messages, 'target_id': 't'})
        assert thread.status_code == 200 and thread.json()['turns_used'] == 2
        assert client.post('/api/moderate', json={'text': 'hello', 'topic': 'gaming'}).status_code == 422
        assert client.post('/api/thread', json={'messages': messages, 'target_id': 'absent'}).status_code == 422
        assert client.get('/').status_code == 200
        assert client.get('/api/benchmarks').status_code == 200
        receipt = {'method': 'ASGI TestClient with actual saved checkpoint; no browser/HTTP socket test',
                   'health': health.json(), 'single_score': single.json()['toxicity_score'],
                   'thread_turns': thread.json()['turns_used'], 'invalid_input_status': 422,
                   'notebook_execution': 'live Jupyter kernel via nbclient' if args.live_kernel else 'sequential in-process', 'cell_timings': timings}
    from contextual.config import write_json
    write_json(ROOT / 'reports/submission_verification.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
