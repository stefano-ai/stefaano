import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import { OpenAIClient } from './openaiClient.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = process.env.PORT || 3000;

const upload = multer({ storage: multer.memoryStorage() });

const client = new OpenAIClient({
  apiKey: process.env.OPENAI_API_KEY,
  safetyMessage:
    process.env.SAFETY_MESSAGE ||
    'Only describe what is visible in the image; do not infer hidden data.',
  requestsPerMinute: Number(process.env.REQUESTS_PER_MINUTE || 20),
});

const app = express();
app.use(express.static(path.join(__dirname, '..', 'public')));

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    rateLimitPerMinute: client.rateLimiter?.requestsPerMinute || null,
  });
});

app.post('/api/describe', upload.single('image'), async (req, res) => {
  const prompt = req.body?.prompt || 'Describe the image';
  const imageBuffer = req.file?.buffer;

  if (!imageBuffer) {
    res.status(400).json({ error: 'Image is required' });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const sendEvent = (data) => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  try {
    await client.streamImageDescription({
      imageBuffer,
      prompt,
      onDelta: (text) => sendEvent({ text }),
    });
    res.write('event: done\n');
    res.write('data: [DONE]\n\n');
    res.end();
  } catch (err) {
    console.error(err);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to process image' });
    } else {
      sendEvent({ error: err.message });
      res.end();
    }
  }
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
