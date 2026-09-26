from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from ..paths import ROOT
_protocol=json.loads((ROOT/"configs/protocol.json").read_text())
SYSTEM_PROMPT=_protocol["system"]
TAG_INTRO=_protocol["tag_intro"]

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, value: Any) -> None:

    def convert(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and (not math.isfinite(item)):
            return None
        return item
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    temporary.write_text(json.dumps(convert(value), ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def tag_prompt(row: dict[str, Any]) -> str:
    return str(row['prompt'])


def feedback_prompt(row: dict[str, Any]) -> str:
    choice = str(row['parsed']['choice'])
    return str(row['feedback'][choice])


@dataclass
class History:
    run_id: int
    messages: list[dict[str, str]]
    tag_rows: list[dict[str, Any]]
    checkpoint_message_indices: dict[str, int]
    checkpoint_feedback_texts: dict[str, str]


def build_history(raw_path: str | Path, expected_trials: int=64) -> History:
    rows = []
    for raw in read_jsonl(raw_path):
        if raw.get('phase') != 'tag':
            continue
        trial = dict(raw['trial'])
        row = dict(raw)
        row.update(trial)
        row['advisor_id'] = str(trial['participant_id'])
        row['prompt'] = str(trial['prompt'])
        row['feedback'] = dict(trial['feedback'])
        rows.append(row)
    rows.sort(key=lambda row: int(row['trial_id']))
    if len(rows) != expected_trials:
        raise ValueError(f'Expected {expected_trials} TAG rows in {raw_path}, got {len(rows)}')
    run_ids = {int(row['run_id']) for row in rows}
    if len(run_ids) != 1:
        raise ValueError(f'Mixed run IDs in {raw_path}: {run_ids}')
    messages: list[dict[str, str]] = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    counts: dict[str, int] = {}
    checkpoints: dict[str, int] = {}
    checkpoint_feedback_texts: dict[str, str] = {}
    for index, row in enumerate(rows):
        prompt = tag_prompt(row)
        if index == 0:
            user_content = TAG_INTRO + '\n\n' + prompt
        else:
            user_content = feedback_prompt(rows[index - 1]) + '\n\n' + prompt
        messages.append({'role': 'user', 'content': user_content})
        messages.append({'role': 'assistant', 'content': str(row['assistant_history_text'])})
        advisor = str(row['advisor_id'])
        counts[advisor] = counts.get(advisor, 0) + 1
        if counts[advisor] == 16:
            checkpoints[advisor] = len(messages) if index < len(rows) - 1 else -1
            checkpoint_feedback_texts[advisor] = feedback_prompt(row)
    messages.append({'role': 'user', 'content': feedback_prompt(rows[-1])})
    for advisor, message_index in list(checkpoints.items()):
        if message_index == -1:
            checkpoints[advisor] = len(messages) - 1
    if sorted(counts.values()) != [16, 16, 16, 16]:
        raise ValueError(f'Unbalanced advisor counts: {counts}')
    if len(checkpoints) != 4:
        raise ValueError(f'Missing checkpoints: {checkpoints}')
    return History(int(next(iter(run_ids))), messages, rows, checkpoints, checkpoint_feedback_texts)


def class_by_advisor(history: History) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in history.tag_rows:
        result[str(row['advisor_id'])] = str(row['advisor_class'])
    return result


def last_truth_by_advisor(history: History) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for row in history.tag_rows:
        result[str(row['advisor_id'])] = bool(row['report_is_truthful'])
    return result


class PresencePenalty(LogitsProcessor):

    def __init__(self, penalty: float, prompt_length: int):
        self.penalty = float(penalty)
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_length:]
        for batch_index in range(generated.shape[0]):
            tokens = torch.unique(generated[batch_index])
            if tokens.numel():
                scores[batch_index, tokens] -= self.penalty
        return scores


def primary_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(type(output))


def replace_primary(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    raise TypeError(type(output))


class ModelAdapter:

    def __init__(self, config: dict[str, Any]):
        model_cfg = config['model']
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg['path'], local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        torch.set_num_threads(4)
        self.model = AutoModelForCausalLM.from_pretrained(model_cfg['path'], dtype=torch.bfloat16, device_map='balanced', max_memory={i: '72GiB' for i in range(torch.cuda.device_count())}, local_files_only=True, attn_implementation='sdpa')
        self.model.eval()
        self.layers = self.model.model.layers
        self.device = next((parameter.device for parameter in self.model.parameters() if parameter.device.type != 'meta'))

    def render(self, messages: list[dict[str, str]], add_generation_prompt: bool=True) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=bool(self.config['model']['enable_thinking']))

    def encode(self, rendered: str, offsets: bool=False) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        encoded = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False, return_offsets_mapping=offsets)
        mapping = encoded.pop('offset_mapping', None)
        return (encoded.to(self.device), mapping)

    def final_logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=False, logits_to_keep=1)
        logits = output.logits
        return logits[:, 0, :] if logits.shape[1] == 1 else logits[:, -1, :]

    def generate(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.config['generation']
        prompt_length = int(inputs['input_ids'].shape[1])
        processors = LogitsProcessorList([PresencePenalty(cfg['presence_penalty'], prompt_length)])
        with torch.inference_mode():
            return self.model.generate(**inputs, max_new_tokens=int(cfg['max_new_tokens']), do_sample=True, temperature=float(cfg['temperature']), top_p=float(cfg['top_p']), top_k=int(cfg['top_k']), repetition_penalty=float(cfg['repetition_penalty']), logits_processor=processors, min_p=0.0, use_cache=True, logits_to_keep=1, pad_token_id=self.tokenizer.pad_token_id)

    def manifest(self) -> dict[str, Any]:
        return {'model_path': self.config['model']['path'], 'layer_count': len(self.layers), 'hidden_size': int(self.model.config.hidden_size), 'vocab_size': len(self.tokenizer), 'model_config_hash': sha256_json(self.model.config.to_dict()), 'chat_template_hash': hashlib.sha256(str(self.tokenizer.chat_template).encode()).hexdigest(), 'torch_version': torch.__version__, 'transformers_version': __import__('transformers').__version__, 'device_map': getattr(self.model, 'hf_device_map', None)}
