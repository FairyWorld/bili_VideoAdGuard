import { describe, expect, it, vi } from 'vitest';

const anthropicCreate = vi.hoisted(() => vi.fn());

vi.mock('@anthropic-ai/sdk', () => ({
  default: class Anthropic {
    messages = { create: anthropicCreate };
  },
}));

import { buildOpenAICompatibleRequest, LLMGateway } from './providers';
import type { LLMInvokePayload } from './types';

const payload: LLMInvokePayload = {
  systemPrompt: 'system prompt',
  userPrompt: 'user prompt',
  responseFormat: 'text',
  maxTokens: 512,
};

describe('buildOpenAICompatibleRequest', () => {
  it.each(['gpt-5.6-luna', 'gpt-4.1', 'kimi-k2'])('omits optional parameters for %s', (model) => {
    expect(buildOpenAICompatibleRequest(payload, model)).toEqual({
      model,
      messages: [
        { role: 'system', content: 'system prompt' },
        { role: 'user', content: 'user prompt' },
      ],
    });
  });

  it('omits temperature from Anthropic Messages requests', async () => {
    anthropicCreate.mockResolvedValue({
      id: 'message-1',
      content: [{ type: 'text', text: 'response' }],
    });

    await LLMGateway.invoke(payload, {
      provider: 'anthropic',
      baseUrl: 'https://api.anthropic.com',
      apiKey: 'test-key',
      model: 'claude-opus-4-6',
    });

    expect(anthropicCreate).toHaveBeenCalledWith({
      model: 'claude-opus-4-6',
      max_tokens: 512,
      system: 'system prompt',
      messages: [{ role: 'user', content: 'user prompt' }],
    });
  });

  it('omits temperature from native custom fetch requests', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ message: { content: 'response' } }),
    });
    vi.stubGlobal('fetch', fetchMock);

    try {
      await LLMGateway.invoke(payload, {
        provider: 'custom_fetch',
        baseUrl: 'http://localhost:11434',
        apiKey: '',
        model: 'local-model',
      });
    } finally {
      vi.unstubAllGlobals();
    }

    const [, options] = fetchMock.mock.calls[0];
    expect(JSON.parse(options.body)).toEqual({
      model: 'local-model',
      messages: [
        { role: 'system', content: 'system prompt' },
        { role: 'user', content: 'user prompt' },
      ],
      stream: false,
    });
  });
});
