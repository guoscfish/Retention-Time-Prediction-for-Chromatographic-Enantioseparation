"""Isolated, ephemeral Codex CLI text relay using the active local login.

No resume/session history, external tools or scientific files enter the model context.
Every event is retained; any actual tool item rejects the entire call.
"""
import json
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from .common import sha, stable_hash, write_once

DISABLED = ['shell_tool','unified_exec','apps','plugins','multi_agent','browser_use',
            'computer_use','image_generation','view_image','memories','skill_search',
            'hooks','sleep_tool','code_mode_host','workspace_dependencies','unbounded_connection_retries']
SKILLS = ['imagegen','openai-docs','plugin-creator','skill-creator','skill-installer']
TRANSCRIPT_PREFIX = 'Continue the following explicit current-round transcript. Respond to its last user message using only the frozen system instructions and this transcript. Return one JSON object.\n'


def settings():
    cli = shutil.which('codex')
    if not cli:
        raise RuntimeError('Codex CLI missing')
    config = tomllib.loads((Path.home()/'.codex/config.toml').read_text())
    if config.get('model_provider','openai') != 'openai':
        raise RuntimeError('this frozen CLI revision requires the current OpenAI provider')
    return {'model':'gpt-6-sol','reasoning_effort':'high','provider':'openai',
        'base_url':'active OpenAI login through local Codex CLI', 'wire_api':'codex-cli',
        'ephemeral':True,'native_tool_calls_required':0,'automatic_fallback':False,
        'cli_path':cli,'cli_sha256':sha(cli),
        'cli_version':subprocess.check_output([cli,'--version'],text=True).strip(),
        'context_directory':str(Path(tempfile.gettempdir())/'odh-free-llm32-scientist-v1-isolated'),
        'disabled_features':DISABLED,'disabled_system_skills':SKILLS,
        'transcript_prefix_sha256':stable_hash(TRANSCRIPT_PREFIX),
        'served_model_version':'not exposed by CLI; requested model and binary version frozen'}


def arguments(config, directory, prompt_file):
    skill_config='['+','.join('{path='+json.dumps(str(Path.home()/'.codex/skills/.system'/s/'SKILL.md'))+',enabled=false}' for s in SKILLS)+']'
    options=['-m',config['model'],'-c','model_reasoning_effort="high"',
        '-c',f'model_instructions_file={json.dumps(str(prompt_file))}',
        '-c','project_doc_max_bytes=0','-c','approval_policy="never"',
        '-c','web_search="disabled"','-c','mcp_servers={}','-c','skills.config='+skill_config]
    for feature in DISABLED:
        options+=['--disable',feature]
    options+=['--enable','skip_host_skill_discovery']
    return options


def isolated_context(config, system_prompt):
    directory=Path(config['context_directory']);directory.mkdir(exist_ok=True)
    prompt_file=directory/'system.txt'
    if prompt_file.exists() and prompt_file.read_text()!=system_prompt:
        raise RuntimeError('isolated system file drift')
    if not prompt_file.exists():prompt_file.write_text(system_prompt)
    if {p.name for p in directory.iterdir()} != {'system.txt'}:
        raise RuntimeError('isolated working directory contains unexpected files')
    return directory,prompt_file


def audit_events(events):
    messages=[]; usage=None
    for event in events:
        if event['type'] in ('item.started','item.updated','item.completed'):
            item=event['item'];kind=item['type']
            if kind not in ('agent_message','reasoning','error'):
                raise RuntimeError(f'forbidden native item: {kind}')
            if kind=='error' and not any(item.get('message','').startswith(s) for s in
                ['Under-development features enabled:','Code Mode is unavailable because code-mode host is disabled.']):
                raise RuntimeError('CLI error item; reject response')
            if kind=='agent_message' and event['type']=='item.completed':messages.append(item['text'])
        elif event['type']=='turn.completed':usage=event.get('usage')
        elif event['type'] not in ('thread.started','turn.started'):
            raise RuntimeError(f'unexpected CLI event: {event["type"]}')
    if len(messages)!=1 or usage is None:
        raise RuntimeError('CLI must complete exactly one text response')
    return messages[0],usage


def freeze_context(config, system_prompt, path):
    directory,prompt_file=isolated_context(config,system_prompt)
    cmd=[config['cli_path'],'--no-daemon','-C',str(directory),*arguments(config,directory,prompt_file),
         'debug','prompt-input','ISOLATION_AUDIT_ONLY']
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    if result.returncode:raise RuntimeError('CLI context audit failed')
    raw=json.loads(result.stdout)
    context=[{'role':m['role'],'content':m['content']} for m in raw]
    text=json.dumps(context)
    if '<skills_instructions>' in text or 'Approved command prefixes' in text:
        raise RuntimeError('injected skills/rules violate isolation')
    if any(marker in text for marker in ['Retention-Time-Prediction','studies/active_learning','<user_instructions>']):
        raise RuntimeError('repository or personal context entered isolated CLI prompt')
    value={'system_prompt_sha256':stable_hash(system_prompt),'cli_config_sha256':stable_hash(config),
           'context':context,'context_sha256':stable_hash(context),
           'note':'Generic CLI permissions/collaboration policy and empty cwd metadata only; no skills, project instructions, scientific artifacts or previous sessions.'}
    write_once(path,value)
    return value


def call(messages, config):
    if config!=settings():raise RuntimeError('CLI configuration drift')
    directory,prompt_file=isolated_context(config,messages[0]['content'])
    cmd=[config['cli_path'],'--no-daemon','exec','--ignore-user-config','--ignore-rules',
         '--ephemeral','--skip-git-repo-check','--json','--sandbox','read-only',
         '--cd',str(directory),*arguments(config,directory,prompt_file),'-']
    result=subprocess.run(cmd,input=TRANSCRIPT_PREFIX+json.dumps(messages[1:],ensure_ascii=False),
                          capture_output=True,text=True,timeout=900)
    if result.returncode:
        raise RuntimeError(f'CLI returned {result.returncode}; no selection accepted')
    events=[json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    answer,usage=audit_events(events)
    return answer,{'requested_model':config['model'],'served_model_version':config['served_model_version'],
        'provider':config['provider'],'cli_version':config['cli_version'],'native_tool_calls':0,
        'usage':usage,'events':events,'stderr_sha256':stable_hash(result.stderr),
        'system_prompt_sha256':stable_hash(messages[0]['content']),
        'explicit_transcript_sha256':stable_hash(messages[1:]),'ephemeral':True}
