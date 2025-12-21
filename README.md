# GPT-4o Image Describer

A minimal Express app that streams GPT-4o descriptions for uploaded JPEGs. Requests are rate-limited and retried with backoff, and every prompt is prefixed with a safety message to avoid speculative answers.

## Project structure
- `src/openaiClient.js`: OpenAI client wrapper that prefixes prompts, enforces rate limits, and streams text deltas.
- `src/rateLimiter.js`: Sliding-window budget tracker for requests per minute.
- `src/server.js`: Express server with SSE streaming endpoint for JPEG uploads.
- `public/`: Lightweight web UI (`index.html`, `app.js`, `styles.css`) for uploading an image and viewing streamed responses.

## Setup
1. Install dependencies:
   ```bash
   npm install
   ```
2. Create a `.env` file with your credentials:
   ```bash
   OPENAI_API_KEY=sk-...
   REQUESTS_PER_MINUTE=20 # optional
   SAFETY_MESSAGE="Only describe what is visible in the image; do not infer hidden data."
   ```
3. Start the server:
   ```bash
   npm start
   ```
4. Open http://localhost:3000 and submit a JPEG plus prompt. Streamed text appears incrementally in the UI.
