const form = document.getElementById('describe-form');
const responseBox = document.getElementById('response');
const promptInput = document.getElementById('prompt');
const imageInput = document.getElementById('image');

const parseEventStream = async (response, onMessage) => {
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
      if (!rawEvent.trim()) continue;

      const lines = rawEvent.split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const data = trimmed.slice(5).trim();
        if (data === '[DONE]') return;
        try {
          onMessage(JSON.parse(data));
        } catch (e) {
          console.error('Failed to parse event', e, data);
        }
      }
    }
  }
};

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = imageInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('prompt', promptInput.value);
  formData.append('image', file);

  responseBox.textContent = '';
  form.querySelector('button').disabled = true;

  try {
    const res = await fetch('/api/describe', {
      method: 'POST',
      body: formData,
      headers: {
        Accept: 'text/event-stream',
      },
    });

    if (!res.ok || !res.body) {
      throw new Error('Request failed');
    }

    await parseEventStream(res, ({ text, error }) => {
      if (error) {
        responseBox.textContent += `\nError: ${error}`;
      }
      if (text) {
        responseBox.textContent += text;
      }
    });
  } catch (err) {
    responseBox.textContent = err.message;
  } finally {
    form.querySelector('button').disabled = false;
  }
});
