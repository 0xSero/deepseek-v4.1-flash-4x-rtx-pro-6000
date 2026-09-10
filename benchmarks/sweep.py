"""Real streaming inference, emitted token IDs, common-window speed evidence."""
import concurrent.futures
import json
import os
import hashlib
from pathlib import Path
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, str(Path(os.environ.get('MODEL_PATH','/models/DeepSeek-V4.1-Flash'))/'encoding'))
from encoding import encode_messages

work = Path(os.environ.get('STATE_PATH','/state'))
output = work/os.environ.get('SWEEP_OUTPUT', 'speed-sweep.jsonl')
manifest_hash = hashlib.sha256((work/'launch.json').read_bytes()).hexdigest()
headers = {'Content-Type':'application/json',
           'Authorization':'Bearer '+(work/'api-key').read_text().strip()}
def run_one(index, concurrency):
    nonce = uuid.uuid4().hex
    instruction = ('Write a complete Python module implementing an LRU cache with a doubly '
        'linked list and dictionary. Include get, put, delete, iteration, resize, clear, '
        'invariant validation and detailed docstrings. Then provide ten usage examples. '
        'Return code only. Implement all methods fully.')
    prompt = encode_messages([{'role':'user', 'content':nonce+'\n'+instruction}], thinking_mode='chat')
    payload = {'text':prompt, 'stream':True, 'sampling_params':{'temperature':0,
        'max_new_tokens':4096}, 'return_logprob':False}
    record = dict(index=index, concurrency=concurrency, nonce=nonce, manifest_sha256=manifest_hash,
        prompt=prompt, output_budget=4096, stream_events=[])
    start = time.monotonic()
    try:
        request = urllib.request.Request('http://127.0.0.1:8010/generate',
            data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=900) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    break
                event = json.loads(data)
                record['stream_events'].append({'seconds':time.monotonic()-start, **event})
        events = record['stream_events']
        record['elapsed_s'] = time.monotonic()-start
        assert events, 'No generated events'
        final = events[-1]
        record['meta_info'] = final.get('meta_info', {})
        # Pinned runtime uses cumulative output_ids by default. Refuse to
        # compute a window if the actual stream does not have this contract.
        previous = []
        for event in events:
            ids = event.get('output_ids', [])
            assert ids[:len(previous)] == previous, 'Stream is not cumulative'
            previous = ids
        record['emitted_tokens'] = len(previous)
        def at(token):
            return next((e['seconds'] for e in events if len(e.get('output_ids', [])) >= token), None)
        record['ttft_s'] = at(1)
        left, right = at(129), at(641)
        if left is not None and right is not None and right > left:
            record['window_129_641_tps'] = 512/(right-left)
        record['finish_reason'] = record['meta_info'].get('finish_reason')
    except Exception as error:
        record['error'] = str(error)
        if hasattr(error, 'read'):
            record['error_body'] = error.read().decode()
    return record

for concurrency in (1, 2, 4, 8):
    wave_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(lambda i: run_one(i, concurrency), range(concurrency)))
    elapsed = time.monotonic()-wave_start
    with output.open('a') as handle:
        for record in records:
            handle.write(json.dumps(record)+'\n')
    summary = dict(concurrency=concurrency, elapsed_s=elapsed,
        errors=[r.get('error') for r in records if r.get('error')],
        tokens=[r.get('emitted_tokens') for r in records],
        ttft_s=[r.get('ttft_s') for r in records],
        matched_window_tps=[r.get('window_129_641_tps') for r in records],
        finish_reasons=[r.get('finish_reason') for r in records])
    print(json.dumps(summary), flush=True)
    if summary['errors']:
        raise RuntimeError('Failed wave; inspect before increasing concurrency')
