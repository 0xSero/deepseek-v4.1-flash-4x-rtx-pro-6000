"""Recompute delivered throughput while every stream in a wave overlaps."""
import json
from pathlib import Path
import statistics
import sys

folder = Path(sys.argv[1])
results = []
for path in sorted(folder.glob('*-c*.json'), key=lambda p: tuple(map(int,p.stem.split('-c')))):
    records = json.loads(path.read_text())
    result = {'input_tokens':int(path.stem.split('-c')[0]),'concurrency':len(records)}
    if any(r.get('error') for r in records):
        result['status'] = 'request_failure'
        results.append(result)
        continue
    timelines = []
    for record in records:
        previous = 0
        timeline = []
        for event in record['events']:
            count = event['count']
            assert count >= previous
            if count > previous:
                timeline.append((event['seconds'],count,count-previous))
            previous = count
        assert previous == len(record['output_ids'])
        assert all(a[0]<=b[0] for a,b in zip(timeline,timeline[1:]))
        timelines.append(timeline)
    if any(not t for t in timelines):
        result['status'] = 'no_output'
        results.append(result)
        continue
    start = max(t[0][0] for t in timelines)
    end = min(t[-1][0] for t in timelines)
    result.update(start_s=start,end_s=end,duration_s=max(0,end-start))
    if end <= start:
        result['status'] = 'no_all_stream_overlap'
    else:
        counts = [sum(delta for timestamp,_,delta in t if start < timestamp <= end) for t in timelines]
        rates = [count/(end-start) for count in counts]
        gaps = [b[0]-a[0] for t in timelines for a,b in zip(t,t[1:]) if start<=a[0]<b[0]<=end]
        result.update(status='measured' if end-start>=2 else 'short_window',
            delivered_tokens_per_request=counts,decode_tps_per_request=rates,
            aggregate_decode_tps=sum(rates),median_decode_tps=statistics.median(rates),
            inter_burst_gap_samples=len(gaps),
            max_inter_burst_gap_s=max(gaps) if gaps else None,
            finish_reasons=[r.get('finish_reason') for r in records])
    results.append(result)
(folder/'common-window.json').write_text(json.dumps(results,indent=2))
lines = ['# Simultaneous delivered decode throughput','',
    'Counts actual emitted token IDs in one shared wall-clock interval: after the last stream starts emitting and before the first stream stops emitting. This includes scheduling stalls and speculative bursts. Windows under two seconds are flagged. No overlap means this burst cannot establish throughput at the requested simultaneous concurrency.','',
    '| Input tokens | C | Shared seconds | Aggregate tok/s | Median tok/s/request | Status |',
    '|---:|---:|---:|---:|---:|---|']
for r in results:
    def fmt(key):
        return f'{r[key]:.2f}' if r.get(key) is not None else '—'
    lines.append(f"| {r['input_tokens']} | {r['concurrency']} | {fmt('duration_s')} | {fmt('aggregate_decode_tps')} | {fmt('median_decode_tps')} | {r['status']} |")
(folder/'COMMON-WINDOW.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
