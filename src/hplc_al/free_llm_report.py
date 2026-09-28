"""Validation-only reporting: never open historical test reports or test truth."""
import csv
import itertools

import numpy as np
from rdkit import Chem, DataStructs

from .common import read_json, sha, write_once
from .free_llm_runner import (STUDY, SOURCE_STUDY, runtime, prepare, state, evaluate,
                             verify_record, verify_protected)
from .free_llm_scientist import BUDGETS, METHOD, SEED, opaque
from .llm_catalog import metadata
from .protocol import RestrictedLabelStore, role_ids


def distribution(values):
    a = np.asarray(values)
    return dict(zip(['min','q25','median','q75','max'], map(float,np.quantile(a,[0,.25,.5,.75,1]))))


def diagnostics(round_index, protocol):
    directory = runtime() / f'round_{round_index}'
    selection = read_json(directory / 'selection.json')
    chosen, pool = selection['selected'], selection['unlabeled_ids']
    features, fingerprints = metadata(pool)
    nonstereo = {i: Chem.MolToSmiles(Chem.MolFromSmiles(features[i]['smiles']), isomericSmiles=False) for i in chosen}
    pairs = {'stereo_contrast':[], 'same_scaffold_contrast':[], 'exact_identity_condition_contrast':[],
             'near_identity_condition_contrast':[]}
    similarities = []
    for i,j in itertools.combinations(chosen,2):
        a,b = features[i],features[j]
        similarity = DataStructs.TanimotoSimilarity(fingerprints[i],fingerprints[j])
        similarities.append(similarity)
        if nonstereo[i] == nonstereo[j] and a['smiles'] != b['smiles']:
            pairs['stereo_contrast'].append([opaque(i),opaque(j)])
        if a['scaffold'] and a['scaffold'] == b['scaffold'] and a['smiles'] != b['smiles']:
            pairs['same_scaffold_contrast'].append([opaque(i),opaque(j)])
        if (a['ipa_fraction'],a['flow']) != (b['ipa_fraction'],b['flow']):
            if a['identity'] == b['identity']:
                pairs['exact_identity_condition_contrast'].append([opaque(i),opaque(j)])
            elif similarity >= .85:
                pairs['near_identity_condition_contrast'].append([opaque(i),opaque(j)])
    with np.load(directory / 'model_state.npz') as saved:
        pos = {i:j for j,i in enumerate(saved['ids'].tolist())}
        coverage = np.array([saved['coverage'][pos[i]] for i in chosen])
        selected_pred = np.array([saved['predictions'][pos[i]] for i in chosen])
        all_pred = np.array([saved['predictions'][pos[i]] for i in pool])
    width = selected_pred[:,2]-selected_pred[:,0]
    width_cut = float(np.quantile(all_pred[:,2]-all_pred[:,0],.9))
    low,high = map(float,np.quantile(all_pred[:,1],[.1,.9]))
    shadow = set(read_json(directory / 'lcmd_shadow.json')['selected'])
    baseline = set(read_json(SOURCE_STUDY / f'runtime/seed_{SEED}/raw_gradient_lcmd/round_{round_index}/selection.json')['selected'])
    def overlap(other):
        return {'intersection':len(set(chosen)&other), 'jaccard':len(set(chosen)&other)/len(set(chosen)|other)}
    result = {'round':round_index, 'pair_counts':{k:len(v) for k,v in pairs.items()},
        'participating_point_counts':{k:len({i for pair in v for i in pair}) for k,v in pairs.items()},
        'pairs':pairs, 'high_coverage_count':int((coverage>=.9).sum()),
        'high_width_count':int((width>=width_cut).sum()), 'width_threshold':width_cut,
        'prediction_extreme_count':int(((selected_pred[:,1]<=low)|(selected_pred[:,1]>=high)).sum()),
        'coverage_percentile_distribution':distribution(coverage), 'q_width_distribution':distribution(width),
        'unique_isomeric_identities':len({features[i]['identity'] for i in chosen}),
        'unique_scaffolds_including_empty':len({features[i]['scaffold_group'] for i in chosen}),
        'empty_scaffold_points':sum(not features[i]['scaffold'] for i in chosen),
        'pairwise_similarity_distribution':distribution(similarities),
        'same_state_lcmd32_overlap':overlap(shadow), 'historical_lcmd32_overlap':overlap(baseline)}
    write_once(directory / 'diagnostics.json',result)
    return result


def report():
    protocol,partition = prepare()
    verify_record(runtime() / 'complete.json')
    state(partition,2)
    valid = role_ids(partition,'validation')
    truth = RestrictedLabelStore(partition,STUDY/'label_access_audit.csv','report_validation').reveal(valid,'validation')
    scale = protocol['frozen_l333_target_population_sd']
    rows = []
    for method in [METHOD,'random','raw_gradient_lcmd']:
        for r,budget in enumerate(BUDGETS):
            if r == 0: source = SOURCE_STUDY / f'shared/seed_{SEED}/fit'
            elif method == METHOD: source = runtime()/f'round_{r-1}/fit'
            else: source = SOURCE_STUDY/f'runtime/seed_{SEED}/{method}/round_{r}/fit'
            record = read_json(source/'fit.json')
            result = evaluate(source,record,valid,truth,scale,budget)
            rows.append({'method':method,**result})
    aulc = []
    for method in [METHOD,'random','raw_gradient_lcmd']:
        group = [r for r in rows if r['method']==method]
        raw = float(np.trapz([r['nrmse'] for r in group],BUDGETS))
        aulc.append({'method':method,'interval':'333–397','raw_nrmse_area':raw,'mean_nrmse':raw/64})
    results=STUDY/'results';results.mkdir(exist_ok=True)
    for name,records in [('validation_metrics',rows),('partial_aulc',aulc)]:
        write_once(results/f'{name}.json',records)
        path=results/f'{name}.csv'
        if not path.exists():
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    diags=[diagnostics(r,protocol) for r in (0,1)]
    texts=[ '# Free-LLM32 阶段一报告：seed 1525，L333 → L397',
        '\n本报告仅使用 validation；只完成两次 acquisition。所有科学机制描述均为 LLM 提出的假设，不是真实机制结论。',
        '\n## 实现与边界',
        f'独立 `{METHOD}`：pending=0，由模型决定全部32点，保留旧16+16实现。代码审计见 AUDIT.md。',
        '输入包括 canonical isomeric SMILES、CIP、Murcko scaffold、官能团、理化描述符、IPA/flow、q10/center/q90/width和原始梯度空间最近已标记点距离百分位。center由MSE训练，不声称为q50；width不是校准epistemic uncertainty，coverage不是真误差。无任何选择配额。',
        '代码确定性拼接本trajectory最近三轮原始批次、真实响应、测量前冻结预测及误差、最高误差观测、历史假设状态和未决问题。每轮新上下文；同轮relay保留完整已发送页。假设更新由同一个实验规划调用明确给出，不由摘要LLM生成。',
        'selector只接收显式packet与allowlist查询。selection → SHA256 seal → Git commit → RestrictedLabelStore.commit_selection → reveal → feedback → scratch fit。逻辑标签边界不是OS隔离；使用单独冻结的Codex CLI transport revision 2、ephemeral新上下文；关闭技能、外部工具、历史规则，审计全部事件并强制native calls=0。原Responses认证失败版本保留；CLI不暴露实际served model快照版本，也不宣称验证其底层wire store标志。validation仅供训练checkpoint选择与独立评价。',
        f"冻结预算：{protocol['limits']}。模型 `{protocol['llm']['model']}`，effort `{protocol['llm']['reasoning_effort']}`；provider由当前配置读取：`{protocol['llm']['base_url']}`。具体served model/version与usage见每轮llm/turn*.json。",
        'resume验证协议、代码、复用文件、请求哈希、selection/feedback/fit文件；已有selection不调用LLM，完成fit不重训；无回执但已有请求intent则fail closed。manifest逐步记录状态，complete须绑定两轮完成的manifest。',
        f"冻结L333 population SD = **{scale:.12f}**。训练器内动态分母NRMSE不用于本报告。基线split/L333/checkpoint/seed/training代码及配置/预算已逐项验证并冻结复用hash；Random32和Raw Gradient-LCMD32均未重训。",
        '\n## Validation 性能',
        '| Method | L | RMSE | MAE | R² | fixed NRMSE |','|---|---:|---:|---:|---:|---:|']
    for r in rows: texts.append(f"| {r['method']} | {r['budget']} | {r['rmse']:.6f} | {r['mae']:.6f} | {r['r2']:.6f} | {r['nrmse']:.6f} |")
    texts += ['\n**Exploratory partial AULC 333–397**：梯形积分，仅此区间；mean=area/64。不可与333–429/525完整区间混用。',
              '| Method | NRMSE area | Mean NRMSE |','|---|---:|---:|']
    for a in aulc: texts.append(f"| {a['method']} | {a['raw_nrmse_area']:.6f} | {a['mean_nrmse']:.6f} |")
    texts += ['\n## 逐轮科学行为与选点诊断']
    for r,d in enumerate(diags):
        directory=runtime()/f'round_{r}';s=read_json(directory/'selection.json');f=read_json(directory/'feedback.json')
        response=s['response'];receipt=read_json(directory/'llm/selection.json')
        texts += [f'\n### Round {r}: L{BUDGETS[r]} → L{BUDGETS[r+1]}',
            f"调用{receipt['calls']}次、query {receipt['queries']}次、候选已浏览{len(receipt['viewed'])}点、选择32点。",
            f"**Batch strategy:** {response['batch_strategy']}",f"**Rationale:** {response['batch_rationale']}",
            f"**Feedback interpretation (LLM):** {response['feedback_interpretation']}",
            f"新hypotheses={len(response['hypotheses'])}；previous updates={len(response['previous_hypothesis_updates'])}。"]
        for u in response['previous_hypothesis_updates']:
            texts.append(f"- 旧假设 `{u['id']}` → **{u['status']}**：{u['reason']} Supporting={u['supporting_observations']}; contradicting={u['contradicting_observations']}。")
        obs={o['id']:o for o in f['observations']}
        for h in response['hypotheses']:
            linked=[obs[i] for i in h['candidate_ids']]
            errors=[o['signed_error'] for o in linked]
            texts.append(f"- 新假设 `{h['id']}`：{h['claim']} Alternative: {h['alternative']}；预期误差符号={h['expected_sign']}；关联实验={h['candidate_ids']}；实际冻结signed errors={[round(e,4) for e in errors]}。这是方向性观测诊断，不自动证明机制。")
        texts += [f"同状态LCMD32 overlap={d['same_state_lcmd32_overlap']}；历史独立LCMD轨迹同轮overlap={d['historical_lcmd32_overlap']}（round1模型和候选池已分叉）。",
            f"对照pair数={d['pair_counts']}；参与点数={d['participating_point_counts']}。stereo contrast不自动等同严格enantiomer pair；pair可重叠，不相加当作32点分配。",
            f"高coverage(≥0.9)={d['high_coverage_count']}/32；高q-width(本U top10%)={d['high_width_count']}/32；prediction extremes(本U两端各10%)={d['prediction_extreme_count']}/32。",
            f"选点coverage百分位分布={d['coverage_percentile_distribution']}；q-width分布={d['q_width_distribution']}。",
            f"不同isomeric identities={d['unique_isomeric_identities']}，scaffolds(含空组)={d['unique_scaffolds_including_empty']}，空scaffold点={d['empty_scaffold_points']}。批内Morgan相似度分布={d['pairwise_similarity_distribution']}。",
            f"未决问题：{response['unresolved_questions']}"]
    texts += ['\n## 解释与下一步',
        '阶段一只能提供单seed、两个批次的探索信号，不能证明LLM化学知识的因果增益。既有validation被用于checkpoint选择，且此cohort存在历史研究使用；不是独立外部验证。',
        'Round1选择前可观察Round0实验反馈并修订假设；Round1实验回来后只记录实际证据，不启动Round2 LLM，因此没有第二批反馈后的LLM修订。必须保留这一闭环长度限制。',
        '\n详细判断与闭环实例见下方人工审阅补充；所有32个理由、hypothesis links及完整响应见每轮selection.json与feedback.json。',
        '\n候选后续（仅建议，不执行）：继续Free-LLM32到L429；用第二seed检验可重复性；以相同protocol单独运行16+16补充型策略。',
        f'\n保护校验：{verify_protected()}个历史artifact保持原hash。新trajectory无test truth访问；无round2、第二seed、masked或hybrid运行。']
    path=STUDY/'INTERIM_REPORT.md'
    if not path.exists(): path.write_text('\n\n'.join(texts)+'\n')
    print({'report':str(path),'aulc':aulc},flush=True)
