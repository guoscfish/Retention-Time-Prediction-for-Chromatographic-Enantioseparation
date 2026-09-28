import copy
import json

import numpy as np
import pytest

from hplc_al.common import atomic_json, ids_hash, metrics, sha, stable_hash
from hplc_al.free_llm_scientist import (LIMITS, METHOD, SEED, SYSTEM_PROMPT, FreeCatalog,
    build_memory, packet, select)
from hplc_al.free_llm_runner import feedback, verify_protected, manifest_update, verify_record
from hplc_al.llm_catalog import Catalog, COMMON_PROMPT, NUMERIC, QUERY_BUDGET
from hplc_al.protocol import RestrictedLabelStore


def inputs():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    features={i:{**{k:float(i+1) for k in NUMERIC},'identity':str(i),'scaffold_group':'group',
        'smiles':'C','scaffold':'','functional_groups':[],'stereocenters':[]} for i in range(72)}
    fp={i:generator.GetFingerprint(Chem.MolFromSmiles('C')) for i in features}
    return features,fp,[0,1],list(range(2,72)),{i:np.array([i,i+1,i+2]) for i in features},{i:i/72 for i in features},{0:{'response':1.},1:{'response':2.}}


def catalog():
    return FreeCatalog(*inputs(),salt='1525/free/0',memory=build_memory([],SEED,METHOD))


def valid(c,p):
    c.query({'pool':'candidates','limit':24,'offset':20})
    return {'type':'selection','packet_hash':p['packet_hash'],'choices':[{'id':i,'reason':'probe','role':'exploratory','evidence_ids':[]} for i in sorted(c.viewed)[:32]],
        'hypotheses':[], 'previous_hypothesis_updates':[], 'batch_strategy':'contrast',
        'batch_rationale':'learning','feedback_interpretation':'','unresolved_questions':[]}


def test_free_pending_zero_and_hybrid_paths():
    c=catalog();p=packet(c,c.memory,0)
    assert p['pending_count']==0 and p['pending_lcmd_ids']==[] and c.selection_count==32
    with pytest.raises(ValueError): FreeCatalog(*inputs(),salt='x',memory=c.memory,pending=[2])
    features,fp,l,u,pred,cov,obs=inputs()
    h=Catalog(features,fp,l,u,[2],pred,cov,obs,salt='x',chemical=True)
    assert h.selection_count==16 and len(h.pools['pending'])==1


def test_exact_unique_legal_viewed():
    c=catalog();p=packet(c,c.memory,0);v=valid(c,p)
    assert len(c.validate(v,p['packet_hash']))==32
    for kind in ['short','duplicate','observed','unviewed','wrong_hash']:
        bad=copy.deepcopy(v)
        if kind=='short':bad['choices'].pop()
        elif kind=='duplicate':bad['choices'][-1]=bad['choices'][0]
        elif kind=='observed':bad['choices'][0]['id']=c.to_id[0]
        elif kind=='unviewed':bad['choices'][0]['id']=next(i for i in c.pools['candidates'] if i not in c.viewed)
        else:bad['packet_hash']='bad'
        with pytest.raises(ValueError):c.validate(bad,p['packet_hash'])


def test_candidate_and_packet_no_label_or_evaluation():
    from hplc_al.free_llm_scientist import assert_packet
    c=catalog();p=packet(c,c.memory,0)
    for card in c.pools['candidates'].values():
        assert not {'response','RT','RTv','signed_error','abs_error','truth'}&set(card)
    assert not {'validation','test','scores','validation_metrics'}&set(p)
    for field in ['test','validation_metrics']:
        bad=copy.deepcopy(p);bad[field]=123
        with pytest.raises(ValueError):assert_packet(bad)
    bad=copy.deepcopy(p);bad['initial_cards'][0]['response']=123
    with pytest.raises(ValueError):assert_packet(bad)
    with pytest.raises(ValueError):c.query({'pool':'candidates','sort_by':'response'})
    with pytest.raises(ValueError):c.query({'pool':'observed','ids':[c.to_id[2]]})


def test_frozen_premeasurement_feedback():
    s={'selected':[2], 'round':0,'prediction_before_measurement':{'2':[1.,8.,10.]},'response':{}}
    result=feedback(s,np.array([3.]),{2:{'smiles':'C','flow':1.}})['observations'][0]
    assert result['signed_error']==5 and result['abs_error']==5 and result['premeasurement_center']==8
    assert result['target']=='RTv_mL' and result['output_column']==1


def test_memory_is_local_and_deterministic():
    b={'seed':SEED,'method':METHOD,'round':0,'response':{'hypotheses':[],
        'previous_hypothesis_updates':[],'unresolved_questions':['why?']},'observations':[]}
    assert build_memory([b],SEED,METHOD)==build_memory([b],SEED,METHOD)
    with pytest.raises(ValueError):build_memory([b],2525,METHOD)
    with pytest.raises(ValueError):build_memory([b],SEED,'chemical_llm')


def test_budget_consistency_and_limits():
    c=catalog();p=packet(c,c.memory,0)
    assert LIMITS=={'query_budget':24,'view_budget':480,'call_budget':28,'page_size':24}
    assert c.query_budget==QUERY_BUDGET==p['limits']['query_budget']==24
    assert 'At most 24 queries' in COMMON_PROMPT and 'Limits: 24 queries' in SYSTEM_PROMPT
    for _ in range(24):c.query({'pool':'observed','limit':1})
    with pytest.raises(ValueError):c.query({'pool':'observed','limit':1})
    with pytest.raises(ValueError):catalog().query({'pool':'candidates','limit':25})


def test_stable_display_independent_row_order():
    first=catalog().initial()
    args=list(inputs());args[0]=dict(reversed(list(args[0].items())));args[3]=list(reversed(args[3]))
    c=FreeCatalog(*args,salt='1525/free/0',memory=build_memory([],SEED,METHOD))
    assert first==c.initial()
    c.salt='1525/free/1';assert first!=c.initial()


def test_barrier_freeze_then_commit_reveal(tmp_path):
    source=tmp_path/'data.csv';source.write_text('RT,Speed\n1,1\n2,1\n3,1\n4,1\n')
    partition={'rows':[{'sample_index':i,'role':'l0' if i==0 else 'u0' if i<3 else 'test'} for i in range(4)]}
    store=RestrictedLabelStore(partition,tmp_path/'audit.csv','free',source)
    with pytest.raises(PermissionError):store.reveal([1],'fit')
    with pytest.raises(PermissionError):store.reveal([3],'final_test')
    # Real B32 commit barrier.
    source.write_text('RT,Speed\n'+'1,1\n'*34)
    partition={'rows':[{'sample_index':i,'role':'l0' if i==0 else 'u0'} for i in range(34)]}
    store=RestrictedLabelStore(partition,tmp_path/'audit.csv','free',source)
    selected=list(range(1,33));s={'method':'free','round':0,'L_hash':ids_hash([0]),'U_hash':ids_hash(range(1,34)),
        'selected':selected,'selected_hash':stable_hash(selected)}
    atomic_json(tmp_path/'selection.json',s)
    with pytest.raises(PermissionError):store.reveal(selected,'fit')
    store.commit_selection(tmp_path/'selection.json');assert len(store.reveal(selected,'fit'))==32


def test_resume_zero_extra_llm_calls_and_full_pages(tmp_path):
    c=catalog();p=packet(c,c.memory,0)
    helper=catalog();hp=packet(helper,helper.memory,0);v=valid(helper,hp)
    answers=[{'type':'query','queries':[{'pool':'candidates','limit':24,'offset':20}]},v]
    calls=[]
    def transport(messages,config):
        calls.append(messages)
        if len(calls)==2:
            relay=json.loads(messages[-1]['content']);assert len(relay['query_results'][0]['records'])==24
        return json.dumps(answers[len(calls)-1]),{'native_tool_calls':0}
    saved=select(p,c,tmp_path,{},transport)
    assert len(calls)==2
    c=catalog();p=packet(c,c.memory,0)
    assert select(p,c,tmp_path,{},transport)==saved and len(calls)==2
    # Tampered request refuses replay.
    atomic_json(tmp_path/'request_00.json',{'request_sha256':'tampered'})
    with pytest.raises(RuntimeError):select(p,catalog(),tmp_path,{},transport)


def test_ambiguous_provider_call_fails_closed(tmp_path):
    c=catalog();p=packet(c,c.memory,0)
    atomic_json(tmp_path/'request_00.json',{'request_sha256':'intent'})
    def forbidden(*args):raise AssertionError('must not call')
    with pytest.raises(RuntimeError,match='ambiguous'):select(p,c,tmp_path,{},forbidden)


def test_fixed_nrmse_denominator():
    scale=8.879240547556758
    for pred in [[1,2,3],[5,7,9]]:
        m=metrics(pred,[2,4,8],scale);assert m['nrmse']==m['rmse']/scale


def test_protected_historical_artifacts_unchanged():
    assert verify_protected()>500


def test_manifest_tampering_fails_closed(tmp_path):
    directory=tmp_path/'round_0';directory.mkdir()
    artifact=directory/'feedback.json';atomic_json(artifact,{'response':1})
    manifest_update(directory,files={str(artifact):sha(artifact)},labels_revealed=True)
    assert verify_record(directory/'manifest.json')['labels_revealed']
    atomic_json(artifact,{'response':2})
    with pytest.raises(RuntimeError):verify_record(directory/'manifest.json')


def test_hypothesis_revision_requires_observed_evidence():
    c=catalog();c.memory['previous_hypotheses']=[{'id':'old'}];p=packet(c,c.memory,0);v=valid(c,p)
    with pytest.raises(ValueError):c.validate(v,p['packet_hash'])
    v['previous_hypothesis_updates']=[{'id':'old','status':'unresolved','reason':'no data',
        'supporting_observations':[],'contradicting_observations':[]}]
    assert len(c.validate(v,p['packet_hash']))==32
    v['previous_hypothesis_updates'][0]['supporting_observations']=[v['choices'][0]['id']]
    with pytest.raises(ValueError):c.validate(v,p['packet_hash'])


def test_real_selection_requires_committed_seal(tmp_path):
    import subprocess
    from hplc_al.free_llm_runner import require_git_commit
    directory=tmp_path/'round_0';directory.mkdir()
    atomic_json(directory/'selection_seal.json',{'files':{}})
    with pytest.raises((ValueError,subprocess.CalledProcessError,RuntimeError)):
        require_git_commit(directory)


def test_phase_boundary_no_round2_or_other_seed():
    from hplc_al.free_llm_runner import make_round, advance
    from hplc_al.free_llm_scientist import BUDGETS
    assert SEED==1525 and BUDGETS==[333,365,397]
    for function in [make_round,advance]:
        with pytest.raises(ValueError): function(2)


def test_manifest_completed_fit_is_reused_without_training(tmp_path,monkeypatch):
    import hplc_al.free_llm_runner as runner
    directory=tmp_path/'round_0';directory.mkdir()
    atomic_json(tmp_path/'protocol.json',{})
    atomic_json(directory/'selection_seal.json',{'files':{},'protocol_sha256':sha(tmp_path/'protocol.json')})
    manifest_update(directory,fit_complete=True)
    monkeypatch.setattr(runner,'STUDY',tmp_path)
    monkeypatch.setattr(runner,'runtime',lambda:tmp_path)
    monkeypatch.setattr(runner,'prepare',lambda:({},{}))
    def forbidden(*args,**kwargs): raise AssertionError('completed fit must not be retrained')
    monkeypatch.setattr(runner,'fit',forbidden)
    assert runner.advance(0)['fit_complete']


def test_frozen_protocol_matches_prompt_packet_when_present():
    from hplc_al.free_llm_runner import STUDY
    from hplc_al.common import read_json
    if (STUDY/'protocol.json').exists():
        protocol=read_json(STUDY/'protocol.json')
        assert protocol['limits']==LIMITS
        assert protocol['prompt_sha256']==stable_hash(SYSTEM_PROMPT)
        assert protocol['frozen_l333_target_population_sd']==8.879240547556758
        assert protocol['pending_count']==0 and protocol['batch_size']==32
