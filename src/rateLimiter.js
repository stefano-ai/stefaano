export class RateLimiter {
  constructor({ requestsPerMinute }) {
    this.requestsPerMinute = requestsPerMinute;
    this.timestamps = [];
  }

  async acquire() {
    if (!this.requestsPerMinute) return;

    const now = Date.now();
    this._prune(now);
    while (this.timestamps.length >= this.requestsPerMinute) {
      const earliest = this.timestamps[0];
      const waitFor = Math.max(earliest + 60_000 - now, 0);
      await new Promise((resolve) => setTimeout(resolve, waitFor));
      this._prune(Date.now());
    }
    this.timestamps.push(Date.now());
  }

  _prune(now) {
    while (this.timestamps.length && now - this.timestamps[0] >= 60_000) {
      this.timestamps.shift();
    }
  }
}
