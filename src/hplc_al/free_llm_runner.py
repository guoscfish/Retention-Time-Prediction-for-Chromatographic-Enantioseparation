"""Two-acquisition, validation-only Free-LLM32 experiment with durable audit stages."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .acquisition import lcmd_tp_select
from .common import (ROOT, SOURCE, atomic_json, code_hashes, ids_hash, metrics, read_json,
                     sha, stable_hash, verify_files, write_once)
from .data import load_graphs, predict
from .free_llm_scientist import (BUDGETS, DESCRIPTOR_SCHEMA, LIMITS, METHOD, SEED,
    SELECTION_SCHEMA, SYSTEM_PROMPT, FreeCatalog, build_memory, opaque, packet, select)
from .gradient import extract
from .lcmd_confirmation import _training
from .llm_catalog import coverage_percentile, metadata
from .llm_transport import settings
from .protocol import RestrictedLabelStore, role_ids, transition
from .training import fit, load_model
from .runner import assert_environment

STUDY = ROOT / 'studies/active_learning/odh_free_llm32_scientist_v1'
SOURCE_STUDY = ROOT / 'studies/active_learning/odh_lcmd_confirmation_v2'
TRAJECTORY = f'{SEED}/{METHOD}'


def runtime():
    return STUDY / f'runtime/seed_{SEED}/{METHOD}'


def verify_protected():
    files = read_json(STUDY / 'protected_artifacts.json')
    verify_files(ROOT, files)
    actual = {str(p.relative_to(ROOT)) for d in (ROOT / 'studies/active_learning').iterdir()
              if d.name.startswith(('odh_llm_hybrid_', 'odh_lcmd_confirmation_', 'odh_ivr_hybrid_'))
              for p in d.rglob('*') if p.is_file() and p.name != '.DS_Store'}
    if actual != set(files):
        raise RuntimeError('historical artifact membership changed')
    return len(files)


def bind_files(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def verify_record(path):
    record = read_json(path)
    verify_files(ROOT, record['files'])
    return record


def prepare():
    environment = assert_environment()
    verify_protected()
    if (STUDY / 'protocol.json').exists():
        value = read_json(STUDY / 'protocol.json')
        verify_files(ROOT, value['source_hashes'])
        verify_files(ROOT, value['reuse_hashes'])
        if value['environment'] != environment:
            raise RuntimeError('frozen runtime environment drift')
        if value['llm'] != settings() or value['limits'] != LIMITS:
            raise RuntimeError('frozen transport/budget drift')
        if value['prompt_sha256'] != stable_hash(SYSTEM_PROMPT):
            raise RuntimeError('frozen prompt drift')
        if sha(STUDY / 'protocol.json') != read_json(STUDY / 'protocol_freeze.json')['sha256']:
            raise RuntimeError('protocol freeze mismatch')
        return value, read_json(SOURCE_STUDY / 'splits/partition.json')
    partition = read_json(SOURCE_STUDY / 'splits/partition.json')
    l0, valid = role_ids(partition, 'l0'), role_ids(partition, 'validation')
    source = SOURCE_STUDY / f'shared/seed_{SEED}/fit'
    record = read_json(source / 'fit.json')
    verify_files(source, record['files'])
    contract = read_json(source / 'input.json')
    if contract['config'] != _training(SEED) or contract['labeled'] != l0 or contract['validation'] != valid:
        raise RuntimeError('L333 reuse contract mismatch')
    if sha(SOURCE) != partition['source_sha256']:
        raise RuntimeError('source data drift')
    store = RestrictedLabelStore(partition, STUDY / 'label_access_audit.csv', TRAJECTORY)
    initial = store.reveal(l0, 'fit')
    scale = float(np.std(initial.astype(np.float64), ddof=0))
    source_protocol = read_json(SOURCE_STUDY / 'protocol.json')
    if environment != source_protocol['environment']:
        raise RuntimeError('baseline runtime environment mismatch')
    core_paths = ['src/hplc_al/training.py', 'src/hplc_al/data.py', 'src/hplc_al/gradient.py',
                  'src/hplc_al/deterministic_shuffle.py', 'src/reproduction/official.py',
                  'code/Single_column_prediction.py', 'code/compound_tools.py']
    for name in core_paths:
        if sha(ROOT / name) != source_protocol['source_hashes'][name]:
            raise RuntimeError(f'baseline training/representation code mismatch: {name}')
    reuse = [SOURCE_STUDY / 'protocol.json', SOURCE_STUDY / 'splits/partition.json',
             source / 'fit.json'] + [source / n for n in record['files']]
    for method in ['random', 'raw_gradient_lcmd']:
        for r, budget in enumerate(BUDGETS):
            directory = SOURCE_STUDY / f'runtime/seed_{SEED}/{method}/round_{r}'
            rec = read_json(directory / 'round.json')
            if rec['budget'] != budget or rec['normalization_scale'] != scale or rec['protocol_hash'] != stable_hash(source_protocol):
                raise RuntimeError('baseline budget/protocol/scale mismatch')
            if r == 0 and (rec['checkpoint_hash'] != record['checkpoint_hash'] or rec['labeled_ids'] != l0):
                raise RuntimeError('baseline L333 checkpoint mismatch')
            fit_dir = source if r == 0 else directory / 'fit'
            fit_rec = read_json(fit_dir / 'fit.json')
            verify_files(fit_dir, fit_rec['files'])
            inp = read_json(fit_dir / 'input.json')
            if inp['config'] != _training(SEED) or inp['labeled'] != rec['labeled_ids'] or inp['validation'] != valid:
                raise RuntimeError('baseline fit contract mismatch')
            if fit_rec['initialization_hash'] != record['initialization_hash']:
                raise RuntimeError('baseline initialization mismatch')
            reuse += [directory / 'round.json', fit_dir / 'fit.json'] + [fit_dir / n for n in fit_rec['files']]
            if (directory / 'selection.json').exists():
                reuse.append(directory / 'selection.json')
    src = code_hashes()
    src['scripts/run_hplc_free_llm32.py'] = sha(ROOT / 'scripts/run_hplc_free_llm32.py')
    protocol = {'study': STUDY.name, 'seed': SEED, 'method': METHOD, 'budgets': BUDGETS,
        'batch_size': 32, 'pending_count': 0, 'limits': LIMITS, 'llm': settings(),
        'environment': environment,
        'prompt_sha256': stable_hash(SYSTEM_PROMPT), 'descriptor_schema': DESCRIPTOR_SCHEMA,
        'selection_schema': SELECTION_SCHEMA, 'descriptor_schema_sha256': stable_hash(DESCRIPTOR_SCHEMA),
        'selection_schema_sha256': stable_hash(SELECTION_SCHEMA), 'training': _training(SEED),
        'partition_sha256': stable_hash(partition), 'l333_ids_sha256': ids_hash(l0),
        'l333_checkpoint_sha256': record['checkpoint_hash'],
        'frozen_l333_target_population_sd': scale, 'target': 'RTv=RT*flow; output[:,1] central MSE predictor',
        'selector_information': DESCRIPTOR_SCHEMA + ['authorized observed labels', 'local frozen premeasurement feedback'],
        'test_truth_access_count': 0, 'stop': 'after two acquisitions and fits at L397; no round2 acquisition',
        'diagnostic_definitions': {'high_coverage': '>=0.9 outer-pool nearest-L gradient-distance percentile',
            'high_width': '>=current legal U 90th percentile q90-q10',
            'prediction_extreme': '<=U 10th or >=U 90th percentile center',
            'near_identity': 'nonchiral Morgan Tanimoto >=0.85',
            'stereo_contrast': 'same nonisomeric canonical SMILES, different isomeric SMILES; not necessarily enantiomers',
            'same_scaffold': 'same nonempty Murcko scaffold, different isomeric SMILES',
            'condition_contrast': 'same identity or similarity>=0.85, differing IPA or flow'},
        'source_hashes': src, 'reuse_hashes': bind_files(set(reuse))}
    write_once(STUDY / 'SYSTEM_PROMPT.json', {'prompt': SYSTEM_PROMPT, 'sha256': stable_hash(SYSTEM_PROMPT)})
    write_once(STUDY / 'protocol.json', protocol)
    write_once(STUDY / 'protocol_freeze.json', {'sha256': sha(STUDY / 'protocol.json'),
        'prompt_file_sha256': sha(STUDY / 'SYSTEM_PROMPT.json')})
    return protocol, partition


def checkpoint_source(round_index):
    if round_index == 0:
        return SOURCE_STUDY / f'shared/seed_{SEED}/fit'
    return runtime() / f'round_{round_index-1}/fit'


def state(partition, round_index):
    if round_index not in (0, 1, 2):
        raise ValueError('only two acquisition rounds allowed')
    labeled, unlabeled = role_ids(partition, 'l0'), role_ids(partition, 'u0')
    history = []
    store = RestrictedLabelStore(partition, STUDY / 'label_access_audit.csv', TRAJECTORY)
    for r in range(round_index):
        directory = runtime() / f'round_{r}'
        manifest = verify_record(directory / 'manifest.json')
        if not manifest['fit_complete']:
            raise RuntimeError('previous fit incomplete')
        verify_record(directory / 'selection_seal.json')
        selected = read_json(directory / 'selection.json')['selected']
        store.commit_selection(directory / 'selection.json')
        labeled, unlabeled = transition(labeled, unlabeled, selected)
        history.append(read_json(directory / 'feedback.json'))
    return labeled, unlabeled, history, store


def manifest_update(directory, **changes):
    path = directory / 'manifest.json'
    old = verify_record(path) if path.exists() else {'seed': SEED, 'method': METHOD,
        'round': int(directory.name.split('_')[-1]), 'selection_frozen': False,
        'labels_revealed': False, 'fit_complete': False, 'next_round_started': False, 'files': {}}
    old.update(changes)
    atomic_json(path, old)
    return old


def make_round(round_index):
    if round_index not in (0, 1):
        raise ValueError('hard stop: no acquisition at L397')
    protocol, partition = prepare()
    labeled, unlabeled, history, store = state(partition, round_index)
    directory = runtime() / f'round_{round_index}'
    directory.mkdir(parents=True, exist_ok=True)
    source = checkpoint_source(round_index)
    fit_record = read_json(source / 'fit.json')
    verify_files(source, fit_record['files'])
    features, fingerprints = metadata(sorted(labeled + unlabeled))
    prepared = directory / 'prepared.json'
    if prepared.exists():
        frozen = verify_record(prepared)
        if frozen['checkpoint_hash'] != fit_record['checkpoint_hash'] or frozen['labeled_ids'] != labeled or frozen['unlabeled_ids'] != unlabeled:
            raise RuntimeError('prepared round state drift')
        with np.load(directory / 'model_state.npz') as saved:
            ids, predictions, coverage = saved['ids'].tolist(), saved['predictions'], saved['coverage']
        observations = {int(k): v for k, v in read_json(directory / 'observed.json').items()}
    else:
        if round_index:
            manifest_update(runtime() / f'round_{round_index-1}', next_round_started=True)
        model = load_model(source / fit_record['checkpoint_path'])
        graphs = load_graphs(partition)
        ids = sorted(labeled + unlabeled)
        predictions = predict(model, graphs, ids)
        phi, audit = extract(model, graphs, ids, progress=lambda x: print({'stage': 'gradient', **x}, flush=True))
        positions = {i: j for j, i in enumerate(ids)}
        coverage = coverage_percentile(phi, [positions[i] for i in labeled])
        # LCMD comparator is computed only as a diagnostic, never fed into planner or pending.
        result = lcmd_tp_select(phi[[positions[i] for i in unlabeled]], phi[[positions[i] for i in labeled]], 32)
        lcmd = [unlabeled[int(i)] for i in result.selected_pool_positions]
        write_once(directory / 'lcmd_shadow.json', {'selected': lcmd, 'gradient_audit': audit})
        truth = store.reveal(labeled, 'fit')
        observations = {i: {'response': float(y)} for i, y in zip(labeled, truth)}
        for batch in history:
            for o in batch['observations']:
                i = next(i for i in labeled if opaque(i) == o['id'])
                observations[i].update({k: o[k] for k in ('premeasurement_center', 'signed_error', 'abs_error')})
        np.savez_compressed(directory / 'model_state.npz', ids=ids, predictions=predictions, coverage=coverage)
        write_once(directory / 'observed.json', observations)
        write_once(prepared, {'labeled_ids': labeled, 'unlabeled_ids': unlabeled,
            'checkpoint_hash': fit_record['checkpoint_hash'], 'protocol_hash': sha(STUDY / 'protocol.json'),
            'files': bind_files([directory / n for n in ('model_state.npz','observed.json','lcmd_shadow.json')])})
    memory = build_memory(history, SEED, METHOD)
    catalog = FreeCatalog(features, fingerprints, labeled, unlabeled,
        dict(zip(ids, predictions)), dict(zip(ids, coverage)), observations,
        salt=f'{SEED}/{METHOD}/{round_index}', memory=memory)
    value = packet(catalog, memory, round_index)
    write_once(directory / 'packet.json', value)
    return protocol, directory, catalog, value, labeled, unlabeled


def run_selection(round_index):
    protocol, directory, catalog, value, labeled, unlabeled = make_round(round_index)
    if (directory / 'selection_seal.json').exists():
        verify_record(directory / 'selection_seal.json')
        return read_json(directory / 'selection.json')
    gate = read_json(STUDY / 'test_gate.json')
    dry = read_json(STUDY / 'dry_run.json')
    if gate['status'] != 'PASS' or gate['source_hashes'] != protocol['source_hashes']:
        raise RuntimeError('passing tests for frozen code required before real LLM call')
    if dry['status'] != 'PASSED' or dry['protocol_sha256'] != sha(STUDY / 'protocol.json'):
        raise RuntimeError('passing dry run for frozen protocol required')
    saved = select(value, catalog, directory / 'llm', protocol['llm'])
    chosen = saved['selected']
    predictions = {catalog.to_id[i]: catalog.pools['candidates'][catalog.to_id[i]] for i in chosen}
    selection = {'seed': SEED, 'method': TRAJECTORY, 'arm': METHOD, 'round': round_index,
        'selected': chosen, 'selected_hash': stable_hash(chosen), 'L_hash': ids_hash(labeled),
        'U_hash': ids_hash(unlabeled), 'labeled_ids': labeled, 'unlabeled_ids': unlabeled,
        'packet_hash': value['packet_hash'], 'response': saved['response'],
        'prediction_before_measurement': {str(i): [predictions[opaque(i)][k] for k in
            ('pred_q10','pred_center','pred_q90')] for i in chosen}}
    write_once(directory / 'selection.json', selection)
    paths = [directory / n for n in ('selection.json','packet.json','prepared.json','model_state.npz','observed.json','lcmd_shadow.json')]
    paths += list((directory / 'llm').glob('*.json'))
    seal = {'selection_sha256': sha(directory / 'selection.json'), 'packet_hash': value['packet_hash'],
        'protocol_sha256': sha(STUDY / 'protocol.json'), 'files': bind_files(paths)}
    write_once(directory / 'selection_seal.json', seal)
    manifest_update(directory, selection_frozen=True, selection_hash=sha(directory / 'selection.json'),
        prediction_hash=sha(directory / 'model_state.npz'), files=bind_files(paths + [directory / 'selection_seal.json']))
    return selection


def require_git_commit(directory):
    path = directory / 'selection_seal.json'
    committed = subprocess.run(['git','show',f'HEAD:{path.relative_to(ROOT)}'], cwd=ROOT,
                               capture_output=True, check=True).stdout
    import hashlib
    if hashlib.sha256(committed).hexdigest() != sha(path):
        raise RuntimeError('selection seal must be git committed before revealing labels')
    return subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()


def feedback(selection, truth, features):
    rows = []
    for i, measured in zip(selection['selected'], truth):
        before = selection['prediction_before_measurement'][str(i)]
        error = float(before[1]) - float(measured)
        rows.append({'id': opaque(i), 'candidate_id': opaque(i), **features[i],
            'prediction_before_measurement': before, 'premeasurement_center': before[1],
            'response': float(measured), 'signed_error': error, 'abs_error': abs(error),
            'target': 'RTv_mL', 'output_column': 1})
    return {'seed': SEED, 'method': METHOD, 'round': selection['round'],
            'response': selection['response'], 'observations': rows}


def evaluate(source, record, valid, valid_truth, scale, budget):
    path = source / Path(record['checkpoint_path']).parent / 'validation_predictions.npz'
    with np.load(path) as saved:
        if saved['ids'].tolist() != valid:
            raise RuntimeError('validation prediction alignment drift')
        values = saved['predictions'][:, 1]
    return {'budget': budget, **metrics(values, valid_truth, scale),
            'fixed_denominator': scale, 'prediction_sha256': sha(path), 'checkpoint_sha256': record['checkpoint_hash']}


def advance(round_index):
    if round_index not in (0, 1):
        raise ValueError('hard stop at L397')
    protocol, partition = prepare()
    directory = runtime() / f'round_{round_index}'
    seal = verify_record(directory / 'selection_seal.json')
    if seal['protocol_sha256'] != sha(STUDY / 'protocol.json'):
        raise RuntimeError('selection protocol mismatch')
    manifest = verify_record(directory / 'manifest.json')
    if manifest['fit_complete']:
        return manifest
    commit = require_git_commit(directory)
    labeled, unlabeled, history, store = state(partition, round_index)
    selection = read_json(directory / 'selection.json')
    store.commit_selection(directory / 'selection.json')
    labeled, unlabeled = transition(labeled, unlabeled, selection['selected'])
    if not (directory / 'feedback.json').exists():
        selected_truth = store.reveal(selection['selected'], 'fit')
        features, _ = metadata(selection['selected'])
        write_once(directory / 'feedback.json', feedback(selection, selected_truth, features))
        write_once(directory / 'feedback_seal.json', {'files': bind_files([directory / 'feedback.json']),
            'selection_hash': sha(directory / 'selection.json'), 'commit': commit})
    verify_record(directory / 'feedback_seal.json')
    # A durable feedback file is bound on first write; repeated selection remains prohibited by seal.
    paths = dict(manifest['files'])
    paths.update(bind_files([directory / 'feedback.json', directory / 'feedback_seal.json']))
    manifest_update(directory, labels_revealed=True, selection_commit=commit,
                    feedback_hash=sha(directory / 'feedback.json'), files=paths)
    truth = store.reveal(labeled, 'fit')
    valid = role_ids(partition, 'validation')
    valid_truth = RestrictedLabelStore(partition, STUDY / 'label_access_audit.csv',
                                      f'{TRAJECTORY}/trainer').reveal(valid, 'validation')
    graphs = load_graphs(partition)
    source = directory / 'fit'
    record = fit(graphs, labeled, truth, valid, valid_truth, protocol['training'], source,
                 {'study': STUDY.name, 'seed': SEED, 'method': METHOD, 'round': round_index+1})
    metric = evaluate(source, record, valid, valid_truth,
                      protocol['frozen_l333_target_population_sd'], len(labeled))
    write_once(directory / 'validation.json', metric)
    paths.update(bind_files([source / n for n in record['files']] + [source / 'fit.json', directory / 'validation.json']))
    manifest_update(directory, fit_complete=True, checkpoint_hash=record['checkpoint_hash'],
        validation_prediction_hash=metric['prediction_sha256'], budget_after=len(labeled), files=paths)
    print({'stage': 'advanced', 'budget': len(labeled), 'validation': metric}, flush=True)
    if round_index == 1:
        for r in (0, 1):
            if not verify_record(runtime() / f'round_{r}/manifest.json')['fit_complete']:
                raise RuntimeError('cannot mark incomplete trajectory complete')
        write_once(runtime() / 'complete.json', {'status': 'PHASE1_COMPLETE_L397', 'seed': SEED,
            'method': METHOD, 'acquisitions': 2, 'budget': 397, 'next_round_started': False,
            'test_truth_access_count': 0, 'files': bind_files([runtime() / f'round_{r}/manifest.json' for r in (0,1)])})
    return metric


def dry_run():
    protocol, directory, catalog, value, labeled, unlabeled = make_round(0)
    catalog.query({'pool':'candidates','limit':24,'offset':20})
    ids = sorted(catalog.viewed)[:32]
    response = {'type':'selection','packet_hash':value['packet_hash'],
        'choices':[{'id':i,'reason':'synthetic validation only','role':'exploratory','evidence_ids':[]} for i in ids],
        'hypotheses':[], 'previous_hypothesis_updates':[], 'batch_strategy':'synthetic dry run',
        'batch_rationale':'contract verification only', 'feedback_interpretation':'', 'unresolved_questions':[]}
    selected = catalog.validate(response, value['packet_hash'])
    transition(labeled, unlabeled, selected)
    write_once(STUDY / 'dry_run.json', {'status':'PASSED', 'pending_count':0, 'selected_count':len(selected),
        'legal_and_viewed':True, 'real_llm_calls':0, 'acquisition_labels_revealed':0,
        'packet_sha256':sha(directory / 'packet.json'), 'protocol_sha256':sha(STUDY / 'protocol.json'),
        'limits':LIMITS, 'protected_artifact_count':verify_protected()})
    print({'status':'DRY_RUN_PASSED', 'limits':LIMITS}, flush=True)
