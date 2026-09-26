"""Query-chunked SDPA for Gemma global attention; all keys and values retained."""


def chunked_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if query.shape[-1] <= 256:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    assert dropout == 0
    qlen, klen = query.shape[-2], key.shape[-2]
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    with sdpa_kernel(SDPBackend.MATH):
        if qlen <= 256:
            return sdpa_attention_forward(module, query, key, value, attention_mask,
                dropout=0.0, scaling=scaling, is_causal=is_causal, **kwargs)
        outputs = []
        for start in range(0, qlen, 256):
            end = min(start + 256, qlen)
            if attention_mask is not None:
                mask = attention_mask[..., start:end, :] if attention_mask.shape[-2] != 1 else attention_mask
            elif causal:
                positions = torch.arange(start, end, device=query.device) + klen - qlen
                mask = (torch.arange(klen, device=query.device)[None, :] <= positions[:, None])[None, None]
            else:
                mask = None
            part, _ = sdpa_attention_forward(module, query[..., start:end, :], key, value, mask,
                dropout=0.0, scaling=scaling, is_causal=False, **kwargs)
            outputs.append(part)
        return torch.cat(outputs, dim=1), None


def install(client, messages):
    import torch
    from types import SimpleNamespace
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
    checks = []
    torch.manual_seed(1701)
    for dtype in (torch.float32, torch.bfloat16):
        for qlen, klen, masked in [(777,777,False), (777,777,True), (257,777,True), (7,777,True), (1,777,True), (1025,2049,True)]:
            q = torch.randn(1, 4, qlen, 512, device="cuda:0", dtype=dtype)
            k = torch.randn(1, 2, klen, 512, device="cuda:0", dtype=dtype)
            v = torch.randn_like(k)
            positions = torch.arange(qlen, device="cuda:0") + klen - qlen
            mask = (torch.arange(klen, device="cuda:0")[None,:] <= positions[:,None])[None,None] if masked else None
            if masked:
                mask[..., :, 15:20] = False
            with sdpa_kernel(SDPBackend.MATH):
                expected, _ = sdpa_attention_forward(module, q, k, v, mask)
            actual, _ = chunked_sdpa(module, q, k, v, mask)
            error = (actual.float() - expected.float()).abs().max().item()
            assert error <= (1e-5 if dtype == torch.float32 else .016), error
            checks.append(dict(dtype=str(dtype), qlen=qlen, klen=klen, masked=masked, max_error=error))
    text = client.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = client.processor(text=text, return_tensors="pt").to(client.model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        original = client.model(**inputs, logits_to_keep=1).logits[:, -1].float().cpu()
    AttentionInterface.register("sdpa", chunked_sdpa)
    with torch.inference_mode():
        modified = client.model(**inputs, logits_to_keep=1).logits[:, -1].float().cpu()
    error = (original - modified).abs().max().item()
    kl = (original.softmax(-1) * (original.log_softmax(-1) - modified.log_softmax(-1))).sum().item()
    same = original.argmax(-1).item() == modified.argmax(-1).item()
    assert error <= .25 and kl <= .001 and same
    client.manifest["attention_execution"] = "global_head512_query256_math_sdpa"
    return dict(tensor_checks=checks, short_logits_max_error=error, short_kl=kl, short_top1_same=same)
