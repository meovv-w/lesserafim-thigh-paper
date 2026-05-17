import json

with open('/mnt/d/HERMES/study/lesserafim-thigh-paper/output/measurements/thigh_results.json') as f:
    data = json.load(f)

print(f'Total measurements: {len(data)}')

members = ['sakura', 'chaewon', 'yunjin', 'kazuha', 'eunchae']
eras = ['antifragile', 'unforgiven', 'easy', 'crazy', 'hot']
overall = {}

for m in members:
    member_data = [d for d in data if d['member'] == m]
    all_vals = [d['avg_circ_cm'] for d in member_data]
    if all_vals:
        overall[m] = {'mean': sum(all_vals)/len(all_vals), 'count': len(all_vals), 'values': all_vals}
    
    for e in eras:
        edata = [d for d in data if d['member'] == m and d['era'] == e]
        if edata:
            vals = [d['avg_circ_cm'] for d in edata]
            print(f'{m}/{e}: {len(edata)} photos, avg={sum(vals)/len(vals):.1f}cm, range={min(vals):.1f}-{max(vals):.1f}')

print()

# Overall ranking
ranked = sorted(overall.items(), key=lambda x: x[1]['mean'], reverse=True)
print('=== OVERALL RANKING ===')
for i, (m, v) in enumerate(ranked, 1):
    print(f'{i}. {m}: {v["mean"]:.1f}cm (n={v["count"]})')
