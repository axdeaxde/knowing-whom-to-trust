"""Provider-specific request/response formats used in the experiment."""
import urllib.request

class NoRedirect(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def build_request(config, messages, max_tokens):
    model, backend = (config['api_model'], config['backend'])
    settings = config['request_settings']
    system = '\n\n'.join((m['content'] for m in messages if m['role'] == 'system'))
    conversation = [m for m in messages if m['role'] != 'system']
    if backend == 'openai_api':
        return ('/v1/chat/completions', dict(model=model, messages=messages, max_completion_tokens=max_tokens, **settings))
    if backend == 'anthropic_api':
        return ('/v1/messages', dict(model=model, system=system, messages=conversation, max_tokens=max_tokens, **settings))
    if backend == 'gemini_api':
        return (f'/v1beta/models/{model}:generateContent', dict(systemInstruction={'parts': [{'text': system}]}, contents=[{'role': 'model' if m['role'] == 'assistant' else 'user', 'parts': [{'text': m['content']}]} for m in conversation], generationConfig=dict(maxOutputTokens=max_tokens, **settings)))
    raise ValueError(backend)


def unpack(config, body):
    backend = config['backend']
    if backend == 'openai_api':
        choice, usage = (body['choices'][0], body['usage'])
        message = choice['message']
        content = message.get('content') or ''
        thinking = bool(message.get('reasoning_content')) or usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0) != 0
        return dict(content=content, response_model=body['model'], input_tokens=usage['prompt_tokens'], output_tokens=usage['completion_tokens'], finish_status=choice['finish_reason'], thinking=thinking, usage=usage, context_truncated=False)
    if backend == 'anthropic_api':
        usage = body['usage']
        content = ''.join((part.get('text', '') for part in body['content'] if part['type'] == 'text'))
        thinking = any((p['type'] in ('thinking', 'redacted_thinking') for p in body['content'])) or usage.get('output_tokens_details', {}).get('thinking_tokens', 0) != 0
        return dict(content=content, response_model=body['model'], input_tokens=sum((usage.get(k, 0) for k in ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'))), output_tokens=usage['output_tokens'], finish_status='stop' if body['stop_reason'] == 'end_turn' else body['stop_reason'], thinking=thinking, usage=usage, context_truncated=bool((body.get('context_management') or {}).get('applied_edits')))
    candidate, usage = (body['candidates'][0], body['usageMetadata'])
    parts = candidate['content']['parts']
    content = ''.join((p.get('text', '') for p in parts if not p.get('thought')))
    thought_tokens = usage.get('thoughtsTokenCount', 0)
    thinking = any((p.get('thought') for p in parts)) or thought_tokens != 0
    accounted = usage['promptTokenCount'] + usage.get('candidatesTokenCount', 0) + thought_tokens
    assert usage['totalTokenCount'] == accounted, 'Unexplained Gemini tokens'
    return dict(content=content, response_model=body['modelVersion'], input_tokens=usage['promptTokenCount'], output_tokens=usage['candidatesTokenCount'] + thought_tokens, finish_status='stop' if candidate['finishReason'] == 'STOP' else candidate['finishReason'], thinking=thinking, usage=usage, context_truncated=False)
