import { RateLimiter } from './rateLimiter.js';

const DEFAULT_SAFETY_MESSAGE =
  'Only describe what is visible in the image; do not infer hidden data.';

export class OpenAIClient {
  constructor({ apiKey, safetyMessage = DEFAULT_SAFETY_MESSAGE, requestsPerMinute = 30 }) {
    this.apiKey = apiKey;
    this.safetyMessage = safetyMessage;
    this.rateLimiter = new RateLimiter({ requestsPerMinute });
  }

  async streamImageDescription({ imageBuffer, prompt, onDelta }) {
    if (!this.apiKey) {
      throw new Error('OPENAI_API_KEY is not configured');
    }

    await this.rateLimiter.acquire();

    const body = {
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: this.safetyMessage },
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            {
              type: 'input_image',
              image_url: {
                mime_type: 'image/jpeg',
                data: imageBuffer.toString('base64'),
              },
            },
          ],
        },
      ],
      max_tokens: 300,
      stream: true,
    };

    const headers = {
      Authorization: `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
    };

    await this.#withRetries(async () => {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorText = await response.text();
        const err = new Error(`OpenAI request failed: ${response.status} ${errorText}`);
        err.status = response.status;
        throw err;
      }

      await this.#consumeStream(response, onDelta);
    });
  }

  async #consumeStream(response, onDelta) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const line of rawEvent.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === '[DONE]') return;
          try {
            const parsed = JSON.parse(payload);
            const delta = parsed?.choices?.[0]?.delta?.content?.[0]?.text;
            if (delta) onDelta(delta);
          } catch (err) {
            // Skip malformed chunk
          }
        }
      }
    }
  }

  async #withRetries(fn, attempts = 3, baseDelay = 250) {
    let lastError;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        return await fn();
      } catch (err) {
        lastError = err;
        const status = err.status || 0;
        const shouldRetry = status === 429 || status >= 500 || status === 408;
        if (!shouldRetry || attempt === attempts) break;
        const delay = baseDelay * 2 ** (attempt - 1);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
    throw lastError;
  }
}
