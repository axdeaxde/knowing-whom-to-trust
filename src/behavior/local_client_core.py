import hashlib
import inspect
import os
import re
import time
from pathlib import Path


def has_active_thought(raw, parsed=None):
    for pattern in (r"<think>(.*?)</think>", r"<\|channel>thought\s*(.*?)<channel\|>"):
        if any(x.strip() for x in re.findall(pattern, raw or "", flags=re.S)):
            return True
    if "<think>" in (raw or "") and "</think>" not in raw:
        return True
    return isinstance(parsed, dict) and any(parsed.get(k) for k in ("reasoning_content", "thinking", "thought"))


class LocalClient:
    def __init__(self, config, max_tokens):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        self.torch, self.config, self.max_tokens = torch, config, max_tokens
        self.backend = config["backend"]
        torch.set_num_threads(8)
        assert torch.cuda.device_count() == 2, "Exactly two assigned GPUs expected"
        path = Path(config["model_path"])
        loader = AutoProcessor if self.backend == "gemma" else AutoTokenizer
        self.processor = loader.from_pretrained(path, local_files_only=True)
        self.tokenizer = self.processor.tokenizer if self.backend == "gemma" else self.processor
        started = time.monotonic()
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
            device_map="balanced", max_memory={i: config["max_memory_per_gpu"] for i in range(2)},
            local_files_only=True, attn_implementation="sdpa")
        self.model.eval()
        assert "logits_to_keep" in inspect.signature(self.model.forward).parameters
        assert all(p.device.type == "cuda" for p in self.model.parameters()), "Unexpected CPU offload"
        hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in ["config.json", "generation_config.json", "chat_template.jinja", "tokenizer_config.json", "model.safetensors.index.json"] if (path / name).exists()}
        self.manifest = dict(model_label=config["label"], model_path=str(path), resolved_path=str(path.resolve()),
            model_class=type(self.model).__name__, torch=torch.__version__, transformers=transformers.__version__,
            dtype=str(self.model.dtype), device_map=self.model.hf_device_map,
            visible_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"), config_hashes=hashes,
            chat_template_sha256=hashlib.sha256(str(self.processor.chat_template).encode()).hexdigest(),
            model_load_seconds=time.monotonic()-started, logits_to_keep=1,
            enable_thinking=False, attention_implementation="sdpa", generation=self.settings())

    def settings(self):
        c = self.config
        result = dict(max_new_tokens=self.max_tokens, do_sample=True, temperature=c["temperature"],
            top_p=c["top_p"], top_k=c["top_k"], repetition_penalty=c["repetition_penalty"],
            use_cache=True, logits_to_keep=1)
        if self.backend == "qwen":
            result["min_p"] = c["min_p"]
        return result

    def complete(self, messages, seed):
        from transformers import LogitsProcessor, LogitsProcessorList
        torch = self.torch
        started = time.monotonic()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if self.backend == "gemma" and "<|think|>" in text:
            raise RuntimeError("Gemma thinking template enabled")
        if self.backend == "qwen" and text.rfind("<think>") > text.rfind("</think>"):
            raise RuntimeError("Qwen open thinking block")
        device = self.model.get_input_embeddings().weight.device
        inputs = (self.processor(text=text, return_tensors="pt") if self.backend == "gemma" else
                  self.tokenizer(text, return_tensors="pt", add_special_tokens=False)).to(device)
        input_length = int(inputs["input_ids"].shape[-1])
        text_config = getattr(self.model.config, "text_config", self.model.config)
        context_limit = getattr(text_config, "max_position_embeddings", None)
        if context_limit is not None and input_length + self.max_tokens > context_limit:
            raise RuntimeError(f"Context budget exceeded: {input_length}+{self.max_tokens}>{context_limit}")
        penalty = self.config["presence_penalty"]

        class GeneratedPresencePenalty(LogitsProcessor):
            def __call__(self, ids, scores):
                for row in range(ids.shape[0]):
                    seen = torch.unique(ids[row, input_length:])
                    scores[row, seen] -= penalty
                return scores

        processors = LogitsProcessorList([GeneratedPresencePenalty()]) if penalty else None
        kwargs = self.settings()
        kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for gpu in range(2):
            torch.cuda.reset_peak_memory_stats(gpu)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs, logits_processor=processors)
        for gpu in range(2):
            torch.cuda.synchronize(gpu)
        generated = output[0, input_length:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        official_parsed = self.processor.parse_response(raw) if self.backend == "gemma" else None
        if self.backend == "gemma":
            if not isinstance(official_parsed, dict) or not isinstance(official_parsed.get("content"), str):
                raise RuntimeError("Unexpected official Gemma parser result")
            content = official_parsed["content"].strip()
        else:
            content = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finish = "stop" if int(generated[-1]) in eos or len(generated) < self.max_tokens else "length"
        return dict(content=content, raw_response=raw, processor_parsed_response=official_parsed,
            rendered_prompt=text, input_tokens=input_length, output_tokens=int(generated.numel()),
            generation_config=dict(kwargs, enable_thinking=False, presence_penalty=penalty,
                attention_execution=self.manifest.get("attention_execution", "sdpa"),
                presence_penalty_scope="newly_generated_tokens" if penalty else "disabled"),
            sampling_seed=seed, sampling_seed_applied=True,
            context_limit=context_limit, context_truncated=False,
            non_thinking_ok=not has_active_thought(raw, official_parsed), finish_status=finish,
            latency_seconds=time.monotonic()-started,
            peak_allocated_gib={str(i): torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)},
            peak_reserved_gib={str(i): torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)})
