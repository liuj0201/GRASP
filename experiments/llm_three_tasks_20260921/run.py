"""Small, local-only feasibility runner. One saved response per case; no retries."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'artifacts/evaluation/llm_three_tasks_20260921'
PORT = 8776
URL = f'http://127.0.0.1:{PORT}'
MODEL = 'Qwen2.5-1.5B-Instruct-Q4_K_M'
TASKS = ('summary', 'sentence')

def http(path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(URL + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def start():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        if http('/health', timeout=2).get('status') == 'ok':
            print(json.dumps({'server': 'already_ready', 'models': http('/v1/models')}), flush=True)
            return
    except OSError:
        pass
    manifest_path = ROOT / 'resources/feedback_value_runtime/runtime_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            'No local llama.cpp server is ready at ' + URL + '. On Windows, run '
            'python resources/feedback_value_runtime/setup_runtime.py first; '
            'on other systems start the server described in docs/REPRODUCIBILITY.md. '
            'Model weights and runtime binaries are not included in this repository.'
        )
    manifest = json.loads(manifest_path.read_text())
    command = list(manifest['suggested_server_args'])
    command[command.index('--port') + 1] = str(PORT)
    command[command.index('--seed') + 1] = '20260921'
    command += ['--alias', MODEL]
    started = time.perf_counter()
    with (OUT / 'server.stdout.log').open('w') as stdout, (OUT / 'server.stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError('Local model server exited; inspect server.stderr.log')
        try:
            if http('/health', timeout=1).get('status') == 'ok':
                receipt = {'pid': process.pid, 'command': command,
                           'model': manifest['model'], 'runtime': manifest['runtime'],
                           'load_seconds': time.perf_counter() - started}
                (OUT / 'server.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
                print(json.dumps({'server': 'ready', 'load_seconds': receipt['load_seconds']}), flush=True)
                return
        except OSError:
            pass
        time.sleep(.5)
    raise TimeoutError('Local model did not become ready within startup deadline')

def run(args):
    modules = {name: importlib.import_module(name + '_task') for name in args.tasks.split(',')}
    cases = [case for module in modules.values() for case in module.prepare() if case['split'] == args.split]
    if args.limit:
        cases = [c for name in modules for c in [x for x in cases if x['task'] == name][:args.limit]]
    directory = OUT / args.label
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'outputs.jsonl'
    previous = {r['id']: r for r in (json.loads(line) for line in output.read_text(encoding='utf-8').splitlines())} if output.exists() else {}
    (directory / 'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases), encoding='utf-8')
    settings = {'model': MODEL, 'temperature': 0, 'seed': 20260921, 'max_tokens': 128,
                'cache_prompt': False, 'stream': False}
    source_hashes = {}
    for name, module in modules.items():
        source = Path(module.__file__)
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = directory / source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError('Source changed; use a new --label instead of mixing versions')
        shutil.copyfile(source, destination)
    (directory / 'protocol.json').write_text(json.dumps({'settings': settings, 'source_hashes': source_hashes,
             'split': args.split, 'tasks': list(modules), 'count': len(cases),
             'scope': 'local feasibility; no model fine-tuning; no cloud judge; raw outputs scored'}, indent=2), encoding='utf-8')
    start()
    with output.open('a', encoding='utf-8') as handle:
        for index, case in enumerate(cases):
            module = modules[case['task']]
            payload = {**settings, 'messages': module.messages(case)}
            request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            if case['id'] in previous:
                if previous[case['id']]['request_sha256'] != request_hash:
                    raise ValueError('Request changed for an existing case; use a new --label')
                continue
            t0 = time.perf_counter()
            response = http('/v1/chat/completions', payload)
            elapsed = time.perf_counter() - t0
            text = response['choices'][0]['message']['content']
            evaluation = module.evaluate(case, text)
            evaluation['checks']['not_truncated'] = response['choices'][0]['finish_reason'] != 'length'
            evaluation['success'] = bool(evaluation['success'] and evaluation['checks']['not_truncated'])
            baselines = {name: {'raw_text': value, 'evaluation': module.evaluate(case, value)}
                         for name, value in module.baselines(case).items()}
            row = {'id': case['id'], 'task': case['task'], 'split': case['split'],
                   'input': case['input'], 'metadata': case.get('metadata', {}),
                   'request': payload, 'request_sha256': request_hash,
                   'raw_text': text, 'evaluation': evaluation, 'latency_seconds': elapsed,
                   'baselines': baselines,
                   'finish_reason': response['choices'][0]['finish_reason'],
                   'usage': response.get('usage'), 'served_model': response.get('model'),
                   'response': response}
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps({'done': index + 1, 'total': len(cases), 'id': case['id'],
                              'success': evaluation['success'], 'seconds': round(elapsed, 2),
                              'text': text}, ensure_ascii=False), flush=True)
    print(json.dumps({'finished': str(output), 'cases': len(cases)}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--label', default='dev_v1')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    start() if args.start else run(args)
