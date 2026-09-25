/*
 * AudioWorklet processor: takes the microphone stream at the AudioContext's
 * native rate (44.1 / 48 kHz), downsamples it to 16 kHz mono with linear
 * interpolation, converts to 16-bit PCM and posts 1024-sample chunks (~64 ms)
 * to the main thread, which forwards them over the WebSocket.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.ratio = sampleRate / this.targetRate; // `sampleRate` is a worklet global
    this.chunkSize = 1024;
    this.inBuf = new Float32Array(0);
    this.pos = 0;
    this.out = new Int16Array(this.chunkSize);
    this.outIdx = 0;
    this.sumSquares = 0;
    this.levelCount = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];

    const merged = new Float32Array(this.inBuf.length + ch.length);
    merged.set(this.inBuf);
    merged.set(ch, this.inBuf.length);
    this.inBuf = merged;

    while (this.pos + 1 < this.inBuf.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      let s = this.inBuf[i] * (1 - frac) + this.inBuf[i + 1] * frac;
      if (s > 1) s = 1;
      else if (s < -1) s = -1;
      this.out[this.outIdx++] = s < 0 ? s * 32768 : s * 32767;
      this.sumSquares += s * s;
      this.levelCount++;
      if (this.outIdx === this.chunkSize) {
        const rms = Math.sqrt(this.sumSquares / this.levelCount);
        this.port.postMessage({ pcm: this.out.buffer, rms }, [this.out.buffer]);
        this.out = new Int16Array(this.chunkSize);
        this.outIdx = 0;
        this.sumSquares = 0;
        this.levelCount = 0;
      }
      this.pos += this.ratio;
    }

    const consumed = Math.floor(this.pos);
    this.inBuf = this.inBuf.slice(consumed);
    this.pos -= consumed;
    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
