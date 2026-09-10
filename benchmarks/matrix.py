"""Measured burst-serving matrix: exact input IDs, cold prefixes, decode window."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.request
import uuid
from tokenizers import Tokenizer

model = Path(os.environ.get('MODEL_PATH','/models/DeepSeek-V4.1-Flash'))
state = Path(os.environ.get('STATE_PATH','/state'))
sys.path.insert(0,str(model/'encoding'))
from encoding import encode_messages
tokenizer = Tokenizer.from_file(str(model/'tokenizer.json'))
manifest = Path(os.environ.get('MANIFEST_PATH',str(state/'launch.json')))
identity = hashlib.sha256(manifest.read_bytes()).hexdigest()
output_budget = int(os.environ.get('OUTPUT_TOKENS','1024'))
ignore_eos = os.environ.get('IGNORE_EOS','0') == '1'
api_base = os.environ.get('API_BASE','http://127.0.0.1:8010').rstrip('/')
assert 641 <= output_budget <= 8192
sizes = [int(x) for x in os.environ.get('PREFILL_SIZES','512,2048,8192,32768,65536,131072,200000,400000').split(',')]
concurrencies = [int(x) for x in os.environ.get('CONCURRENCIES','1,2,4,8').split(',')]
folder = state/('matrix-'+time.strftime('%Y%m%dT%H%M%S'))
folder.mkdir(parents=True)
headers = {'Content-Type':'application/json','Authorization':'Bearer '+(state/'api-key').read_text().strip()}
def encode(text):
    return tokenizer.encode(text,add_special_tokens=False).ids
filler = encode('Reference notes: the cache stores recently accessed entries. '
    'An implementation should maintain ordering, handle replacement and validate its invariants.\n')
instruction = ('\nNow write a complete Python LRU cache module with a doubly linked list and dictionary, '
    'including get, put, delete, iteration, resize, clear, invariant validation, detailed docstrings '
    'and ten usage examples. Return code only. Implement all methods fully.\n<｜Assistant｜></think>')
suffix = encode(instruction)
all_summaries = []

def one(size,index,wave_start):
    nonce = uuid.uuid4().hex
    prefix = encode(encode_messages([dict(role='user',content=nonce+'\nRead these notes.\n')],
        thinking_mode='chat').split('<｜Assistant｜>')[0])
    room = size-len(prefix)-len(suffix)
    assert room >= 0
    ids = prefix+(filler*(room//len(filler)+1))[:room]+suffix
    assert len(ids) == size
    result = dict(index=index,prompt_tokens=size,nonce=nonce,manifest_sha256=identity,
        started_s=time.monotonic()-wave_start,events=[],output_budget=output_budget,ignore_eos=ignore_eos)
    try:
        req = urllib.request.Request(api_base+'/generate',headers=headers,
            data=json.dumps(dict(input_ids=ids,stream=True,
                sampling_params={'temperature':0,'max_new_tokens':output_budget,'ignore_eos':ignore_eos})).encode())
        with urllib.request.urlopen(req,timeout=2400) as response:
            previous = []
            for line in response:
                if not line.startswith(b'data: '): continue
                if line[6:].strip() == b'[DONE]': break
                event = json.loads(line[6:])
                current = event.get('output_ids',[])
                assert current[:len(previous)] == previous, 'Expected cumulative output IDs'
                previous = current
                result['events'].append(dict(seconds=time.monotonic()-wave_start,
                    count=len(current),meta_info=event.get('meta_info',{})))
                result['output_ids'] = current
                result['text'] = event.get('text','')
        result['finished_s'] = time.monotonic()-wave_start
        def at(n):
            return next((e['seconds'] for e in result['events'] if e['count']>=n),None)
        result['first_token_s'] = at(1)
        result['ttft_s'] = at(1)-result['started_s']
        left,right = at(129),at(641)
        result['decode_window_tps'] = 512/(right-left) if left is not None and right is not None and right>left else None
        result['finish_reason'] = result['events'][-1]['meta_info'].get('finish_reason')
    except Exception as error:
        result['error'] = repr(error)
        if hasattr(error,'read'): result['error_body'] = error.read().decode()
    return result

def render():
    lines = ['# Measured inference matrix','',
        'Effective prefill includes queuing and mixed decode work until the last request reaches its first token. '
        'Decode is the median per-request rate over emitted tokens 129–641, including serving stalls. '
        'Concurrency is requested burst size; client overlap records how many streams had emitted tokens and remained unfinished. '
        f'Synthetic repeated notes, unique prefixes, {output_budget:,}-token output budget, ignore_eos={ignore_eos}; this is not a quality test.','',
        '| Input tokens/request | Requested C | Effective prefill tok/s | Median TTFT s | Median decode tok/s/request | Total shared decode tok/s | Shared decode tok/s/request | Shared seconds | Peak client overlap | Success |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in all_summaries:
        def fmt(k): return f'{r[k]:.2f}' if r.get(k) is not None else '—'
        lines.append(f"| {r['input_tokens']} | {r['concurrency']} | {fmt('effective_prefill_tps')} | {fmt('median_ttft_s')} | {fmt('median_decode_tps')} | {fmt('total_decode_tps')} | {fmt('shared_decode_tps_per_request')} | {fmt('shared_decode_seconds')} | {r['peak_client_overlap']} | {r['successes']}/{r['concurrency']} |")
    (folder/'TABLE.md').write_text('\n'.join(lines)+'\n')

print(str(folder),flush=True)
for size in sizes:
    for concurrency in concurrencies:
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            records = list(pool.map(lambda i:one(size,i,start),range(concurrency)))
        (folder/f'{size}-c{concurrency}.json').write_text(json.dumps(records))
        good = [r for r in records if not r.get('error')]
        spans = sorted([(r['first_token_s'],1) for r in good]+[(r['finished_s'],-1) for r in good])
        active=peak=0
        for _,delta in spans:
            active+=delta
            peak=max(peak,active)
        rates = [r['decode_window_tps'] for r in good if r.get('decode_window_tps') is not None]
        summary = dict(input_tokens=size,concurrency=concurrency,successes=len(good),
            peak_client_overlap=peak,manifest_sha256=identity,
            effective_prefill_tps=size*len(good)/max(r['first_token_s'] for r in good) if good else None,
            median_ttft_s=statistics.median(r['ttft_s'] for r in good) if good else None,
            median_decode_tps=statistics.median(rates) if rates else None,
            errors=[r['error'] for r in records if r.get('error')])
        summary.update(total_decode_tps=None,shared_decode_tps_per_request=None,shared_decode_seconds=0)
        if len(good) == concurrency:
            timelines = []
            for record in good:
                previous = 0
                timeline = []
                for event in record['events']:
                    count = event['count']
                    if count > previous:
                        timeline.append((event['seconds'],count-previous))
                    previous = count
                timelines.append(timeline)
            if all(timelines):
                begin = max(t[0][0] for t in timelines)
                end = min(t[-1][0] for t in timelines)
                if end > begin:
                    shared_rates = [sum(n for timestamp,n in t if begin < timestamp <= end)/(end-begin) for t in timelines]
                    summary.update(total_decode_tps=sum(shared_rates),shared_decode_tps_per_request=statistics.median(shared_rates),shared_decode_seconds=end-begin)
        all_summaries.append(summary)
        (folder/'summary.json').write_text(json.dumps(all_summaries,indent=2))
        render()
        print(json.dumps(summary),flush=True)
        if len(good)!=concurrency:
            raise RuntimeError('Failed wave; inspect runtime before continuing')
