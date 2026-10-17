#!/usr/bin/env python3
"""Validate the standalone Joint-240 full-scan extension."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
ledger={}
for line in (P/'CHECKSUMS.sha256').read_text().splitlines():
    digest,rel=line.split('  ',1);assert rel not in ledger;ledger[rel]=digest
actual={p.relative_to(P).as_posix() for p in P.rglob('*') if p.is_file()}
assert actual==set(ledger)|{'CHECKSUMS.sha256'},(actual-set(ledger),set(ledger)-actual)
assert all(sha(P/rel)==digest for rel,digest in ledger.items())
meta=json.loads((P/'metadata/provenance.json').read_text())
assert meta['released_sha256']==sha(P/'data/full_sbs_state_scan.csv.gz')
assert meta['source_rows']==453016 and meta['excluded_matched_feature_keys']==98 and meta['released_rows']==452918
with (P/'metadata/scenario_manifest.csv').open(newline='') as f:scenarios={r['scenario_id'] for r in csv.DictReader(f)}
assert len(scenarios)==240
with (P/'metadata/split_oof_folds.csv').open(newline='') as f:foldmap={r['scenario_id']:int(r['fold']) for r in csv.DictReader(f)}
assert set(foldmap)==scenarios and set(foldmap.values())==set(range(5))
with (P/'metadata/excluded_98_matched_feature_states.tsv').open(newline='') as f:excluded={(r['scenario_id'],int(r['step'])) for r in csv.DictReader(f,delimiter='\t')}
assert len(excluded)==98
ids=set();keys=set();seen_scenarios=set();counts={0:0,1:0};n=0
with gzip.open(P/'data/full_sbs_state_scan.csv.gz','rt',newline='') as f:
    rd=csv.DictReader(f)
    assert len(rd.fieldnames)==127
    features=[x for x in rd.fieldnames if x.startswith('state__')]
    assert len(features)==95
    for r in rd:
        n+=1
        k=(r['scenario_id'],int(r['step']))
        assert k not in keys and k not in excluded
        keys.add(k)
        assert r['state_id'] not in ids;ids.add(r['state_id'])
        assert r['scenario_id'] in scenarios and int(r['fold'])==foldmap[r['scenario_id']]
        assert r['trajectory_policy']=='kv_constrained_online'
        assert r['any_non_sbs_policy_differs'] in ('0','1')
        counts[int(r['any_non_sbs_policy_differs'])]+=1
        assert 1<=int(r['n_distinct_canonical_p6_actions_full'])<=6
        assert all(math.isfinite(float(r[x])) for x in features)
        seen_scenarios.add(r['scenario_id'])
assert n==452918 and counts=={0:444032,1:8886} and seen_scenarios==scenarios
print(json.dumps({'status':'PASS','files':len(actual),'checksummed_files':len(ledger),'states':n,'scenarios':len(scenarios),'disagreement_states':counts[1],'equal_action_states':counts[0],'excluded_ambiguous_keys':len(excluded)},sort_keys=True))
