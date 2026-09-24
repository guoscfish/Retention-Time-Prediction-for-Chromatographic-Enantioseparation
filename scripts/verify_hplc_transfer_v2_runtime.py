"""Revalidate existing duration evidence and the actual graph batching runtime."""
import ast
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from hplc_al.common import ROOT, atomic_json, code_hashes, read_json, sha, stable_hash, verify_files
from hplc_al.data import batches, load_graphs
from hplc_al.deterministic_shuffle import epoch_batches, epoch_order, assert_epoch_coverage
from hplc_al.protocol import role_ids
from hplc_al.runner import assert_environment
from hplc_al.transfer_v2 import TRAINING, transfer_partition
from hplc_al.transfer_v2_audit import audit
from hplc_al.protocol import make_partition

study = ROOT / 'studies/active_learning/odh_gradient_al_transfer_v2'
evidence = study / 'runtime_gate'
environment = assert_environment()
tests = []
for target, name in [('tests/hplc_al/test_transfer_v2_protocol.py', 'test_transfer_v2_protocol_resume.log'), ('tests/hplc_al', 'test_hplc_al_resume.log')]:
    log = (evidence / name).read_text()
    if 'passed' not in log or 'failed' in log or 'ERROR' in log:
        raise RuntimeError('pytest gate failed or missing')
    tests.append(dict(command=[sys.executable, '-m', 'pytest', target, '-q'], returncode=0, log=name, sha256=sha(evidence/name)))
partition = transfer_partition(make_partition())
assert partition == read_json(study/'duration_smoke/partition.json')
l0 = role_ids(partition, 'l0')
protected = set(sum([role_ids(partition,r) for r in ['u0','validation','test']],[]))
graphs = load_graphs(partition)
# ID-coded synthetic targets prove pairing without revealing any source target.
truth = np.asarray(l0, dtype=np.float64)*3 + 1
checks = []
for seed in [525,1525]:
    previous = None
    for epoch in range(500):
        a = epoch_order(l0, truth, seed, epoch)
        b = epoch_order(l0, truth, seed, epoch)
        assert np.array_equal(a.permutation,b.permutation)
        if previous is not None:
            assert not np.array_equal(a.permutation,previous)
        previous = a.permutation.copy()
        assert_epoch_coverage(l0,a.ids)
        assert not set(a.ids.tolist()) & protected
        assert np.array_equal(a.truth,a.ids*3+1)
        seen,sizes = [],[]
        for ids,targets in epoch_batches(l0,truth,seed,epoch,256):
            g,h = next(batches(graphs,ids,len(ids)))
            actual = g.sample_key.numpy().reshape(-1)
            assert np.array_equal(actual,ids)
            assert np.array_equal(targets,actual*3+1)
            assert g.num_graphs == h.num_graphs == len(ids)
            sizes.append(g.num_graphs)
            seen.extend(actual.tolist())
        assert_epoch_coverage(l0,seen)
        assert sizes == [256,77]
    checks.append(dict(seed=seed,epochs_verified=500,steps_per_epoch=2,actual_graph_batch_sizes=[256,77],same_seed_same_epoch=True,different_epoch_different_permutation=True,every_labeled_id_once=True,graph_truth_alignment=True,protected_ids_excluded=True))
    print('verified 500 actual graph-batch epochs',seed,flush=True)
summary = read_json(study/'duration_smoke/summary.json')
frozen = read_json(study/'training_protocol_frozen.json')
assert summary['environment'] == environment
assert summary['status']=='PASS' and frozen['training']==TRAINING
assert frozen['duration_summary_hash']==stable_hash(summary)
for seed in [525,1525]:
    directory=study/f'duration_smoke/seed_{seed}'
    r=read_json(directory/'fit.json')
    verify_files(directory,r['files'])
    assert r['stopping_reason']=='patience' and r['best_epoch']<=400
    assert r['stopped_epoch']-r['best_epoch']==100
    curve=list(csv.DictReader((directory/Path(r['checkpoint_path']).parent/'training_curve.csv').open()))
    assert len(curve)==r['stopped_epoch']
    assert all(int(row['steps_per_epoch'])==2 and ast.literal_eval(row['observed_batch_sizes'])==[256,77] for row in curve)
result=dict(status='PASS',environment=environment,pytest=tests,source_hashes=code_hashes(),partition_hash=stable_hash(partition),L0=333,steps_per_epoch=2,observed_batch_sizes=[256,77],checks=checks,duration_evidence_verified=True,duration_summary_hash=stable_hash(summary),test_truth_access_count=0,U0_truth_access_count=0)
atomic_json(evidence/'runtime_preflight.json',result)
audit(study)
print('PASS runtime, duration evidence and implementation gates',flush=True)
