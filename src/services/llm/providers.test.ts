import { describe, expect, it } from 'vitest';
import { buildOpenAICompatibleRequest } from './providers';
import type { LLMInvokePayload } from './types';

const payload: LLMInvokePayload = {
  systemPrompt: 'system prompt',
  userPrompt: 'user prompt',
  responseFormat: 'text',
  maxTokens: 512,
  temperature: 0.7,
};

describe('buildOpenAICompatibleRequest', () => {
  it('uses GPT-5-compatible parameters for GPT-5 models', () => {
    const request = buildOpenAICompatibleRequest(payload, ' gpt-5.6-luna ');

    expect(request).toMatchObject({
      model: ' gpt-5.6-luna ',
      max_completion_tokens: 512,
    });
    expect(request).not.toHaveProperty('max_tokens');
    expect(request).not.toHaveProperty('temperature');
  });

  it('keeps legacy parameters for non-GPT-5 models', () => {
    const request = buildOpenAICompatibleRequest(payload, 'gpt-4.1');

    expect(request).toMatchObject({
      model: 'gpt-4.1',
      max_tokens: 512,
      temperature: 0.7,
    });
    expect(request).not.toHaveProperty('max_completion_tokens');
  });

  it('omits temperature for Kimi models', () => {
    const request = buildOpenAICompatibleRequest(payload, 'kimi-k2');

    expect(request).toMatchObject({
      max_tokens: 512,
    });
    expect(request).not.toHaveProperty('temperature');
  });
});
