const { createApp } = Vue;
createApp({
  data() {
    return {
      apiBase: window.location.origin,
      phase: "idle", // idle | guide | recording | processing | results
      mode: "",
      errorMsg: "",
      // Session
      sessionId: "",
      chunkSeq: 0,
      chunkCount: 0,
      // Recording
      recordingTime: 0,
      recordingTimer: null,
      latestChunkText: "",
      lastMetrics: null,
      // Audio internals
      audioCtx: null,
      scriptProc: null,
      micStream: null,
      sysStream: null,
      pcmBuf: [],
      pcmLen: 0,
      sendInterval: null,
      chunkQueue: [],
      sending: false,
      // Results
      timeline: [],
      fullText: "",
      // Edit modal
      editOpen: false,
      editIdx: -1,
      editText: "",
      editSpeaker: "",
      // Chunk duration (seconds)
      recChunkSec: 5,     // 실시간 녹음
      fileChunkSec: 30,   // 파일 테스트
      // 실시간 확인 (on: 5초마다 STT 수행, off: 오디오 저장만)
      realtimeStt: true,
      // File test
      showFileTest: false,
      audioFile: null,
      fileBusy: false,
      // DEV: debug raw data
      showDebug: false,
      rawSttItems: [],
      rawDiarSegments: [],
      rawSttFinalItems: [],
    };
  },
  computed: {
    formattedTime() {
      const m = String(Math.floor(this.recordingTime / 60)).padStart(2, "0");
      const s = String(this.recordingTime % 60).padStart(2, "0");
      return `${m}:${s}`;
    },
    speakerList() {
      return [...new Set(this.timeline.map((t) => t.speaker))];
    },
  },
  methods: {
    // ── API ────────────────────────────────────────────────
    async api(path, opts = {}) {
      const res = await fetch(`${this.apiBase}${path}`, opts);
      const text = await res.text();
      let body = {};
      try {
        body = text ? JSON.parse(text) : {};
      } catch {
        body = { raw: text };
      }
      if (!res.ok) {
        const err = new Error(body.detail || body.raw || `HTTP ${res.status}`);
        err.status = res.status;
        throw err;
      }
      return body;
    },
    async createSession() {
      const r = await this.api("/api/session", { method: "POST" });
      this.sessionId = r.session_id;
      this.chunkSeq = 0;
      this.chunkCount = 0;
      this.latestChunkText = "";
      this.lastMetrics = null;
      this.timeline = [];
      this.fullText = "";
    },
    // ── Audio helpers ─────────────────────────────────────
    downsample(buf, fromRate, toRate) {
      if (fromRate === toRate) return new Float32Array(buf);
      const ratio = fromRate / toRate;
      const len = Math.round(buf.length / ratio);
      if (len <= 0) return new Float32Array(0);
      const out = new Float32Array(len);
      for (let i = 0; i < len; i++) {
        const si = i * ratio;
        const lo = Math.floor(si);
        const hi = Math.min(lo + 1, buf.length - 1);
        const f = si - lo;
        out[i] = buf[lo] * (1 - f) + buf[hi] * f;
      }
      return out;
    },
    mergeFloat32(arrays, total) {
      const out = new Float32Array(total);
      let off = 0;
      for (const a of arrays) {
        out.set(a, off);
        off += a.length;
      }
      return out;
    },
    float32ToInt16Bytes(f32) {
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        i16[i] = s < 0 ? s * 32768 : s * 32767;
      }
      return new Uint8Array(i16.buffer);
    },
    // ── Recording ─────────────────────────────────────────
    async startOffline() {
      this.mode = "offline";
      this.errorMsg = "";
      try {
        await this.createSession();
        const mic = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        });
        this.micStream = mic;
        this.setupCapture(mic, null);
        this.phase = "recording";
      } catch (e) {
        this.errorMsg = `녹음 시작 실패: ${e.message}`;
        this.cleanup();
        this.phase = "idle";
      }
    },
    showGuide() {
      this.phase = "guide";
    },
    async startOnline() {
      this.mode = "online";
      this.errorMsg = "";
      try {
        await this.createSession();
        let sys = null;
        try {
          sys = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
          this.sysStream = sys;
        } catch (e) {
          this.errorMsg = "시스템 오디오를 가져올 수 없습니다. 마이크만 사용합니다.";
        }
        const mic = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        });
        this.micStream = mic;
        this.setupCapture(mic, sys);
        this.phase = "recording";
      } catch (e) {
        this.errorMsg = `녹음 시작 실패: ${e.message}`;
        this.cleanup();
        this.phase = "idle";
      }
    },
    setupCapture(micStream, sysStream) {
      const ctx = new AudioContext();
      this.audioCtx = ctx;
      const nativeRate = ctx.sampleRate;
      const targetRate = 16000;
      const chunkSamples = targetRate * this.recChunkSec;
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      this.scriptProc = proc;
      const mixer = ctx.createGain();
      mixer.gain.value = 1.0;
      const micSrc = ctx.createMediaStreamSource(micStream);
      micSrc.connect(mixer);
      if (sysStream) {
        const sysTracks = sysStream.getAudioTracks();
        if (sysTracks.length > 0) {
          const sysAudioStream = new MediaStream(sysTracks);
          const sysSrc = ctx.createMediaStreamSource(sysAudioStream);
          sysSrc.connect(mixer);
        }
      }
      mixer.connect(proc);
      const silent = ctx.createGain();
      silent.gain.value = 0;
      proc.connect(silent);
      silent.connect(ctx.destination);
      this.pcmBuf = [];
      this.pcmLen = 0;
      proc.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const ds = this.downsample(input, nativeRate, targetRate);
        this.pcmBuf.push(new Float32Array(ds));
        this.pcmLen += ds.length;
      };
      this.sendInterval = setInterval(() => {
        if (this.pcmLen >= chunkSamples) {
          const merged = this.mergeFloat32(this.pcmBuf, this.pcmLen);
          const chunk = merged.slice(0, chunkSamples);
          const remainder = merged.slice(chunkSamples);
          this.pcmBuf = remainder.length > 0 ? [new Float32Array(remainder)] : [];
          this.pcmLen = remainder.length;
          const bytes = this.float32ToInt16Bytes(chunk);
          this.enqueueChunk(bytes);
        }
      }, 1000);
      this.recordingTime = 0;
      this.recordingTimer = setInterval(() => {
        this.recordingTime++;
      }, 1000);
      if (ctx.state === "suspended") ctx.resume();
    },
    // ── Chunk queue ───────────────────────────────────────
    enqueueChunk(bytes) {
      this.chunkQueue.push(bytes);
      this.processQueue();
    },
    async processQueue() {
      if (this.sending) return;
      this.sending = true;
      while (this.chunkQueue.length > 0) {
        const bytes = this.chunkQueue.shift();
        await this.sendSingleChunk(bytes);
      }
      this.sending = false;
    },
    async sendSingleChunk(bytes) {
      const seq = this.chunkSeq++;
      const rt = this.realtimeStt ? 1 : 0;
      try {
        const r = await this.api(`/api/session/${this.sessionId}/chunk?seq=${seq}&realtime=${rt}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: bytes,
        });
        if (this.realtimeStt) {
          this.latestChunkText = r.chunk_text || "";
        }
        this.chunkCount = seq + 1;
        if (r.audio_metrics) this.lastMetrics = r.audio_metrics;
      } catch (e) {
        console.error("chunk send error:", e);
        if (e.status === 413 || e.status === 429) {
          this.errorMsg = e.message;
        }
      }
    },
    // ── Stop & Finalize ───────────────────────────────────
    async stopAndFinalize() {
      clearInterval(this.recordingTimer);
      clearInterval(this.sendInterval);
      this.phase = "processing";
      // Flush remaining audio
      if (this.pcmLen > 0) {
        const merged = this.mergeFloat32(this.pcmBuf, this.pcmLen);
        const bytes = this.float32ToInt16Bytes(merged);
        this.enqueueChunk(bytes);
        this.pcmBuf = [];
        this.pcmLen = 0;
      }
      // Wait for queue to drain
      const waitStart = Date.now();
      while ((this.chunkQueue.length > 0 || this.sending) && Date.now() - waitStart < 60000) {
        await new Promise((r) => setTimeout(r, 200));
      }
      // Cleanup audio
      if (this.scriptProc) this.scriptProc.disconnect();
      if (this.audioCtx && this.audioCtx.state !== "closed") {
        try { await this.audioCtx.close(); } catch {}
      }
      this.stopStreams();
      // stop -> background diarization + STT reprocessing
      try {
        await this.api(`/api/session/${this.sessionId}/stop`, { method: "POST" });
        await this._pollUntilDone();
        const r = await this.api(`/api/session/${this.sessionId}/finalize`, { method: "POST" });
        this.timeline = (r.timeline || []).map((t) => ({ ...t }));
        this.fullText = r.full_text || "";
        this.rawSttItems = r.raw_stt_items || [];
        this.rawDiarSegments = r.raw_diar_segments || [];
        this.rawSttFinalItems = r.raw_stt_final_items || [];
        this.phase = "results";
        if (this.timeline.length === 0 && !this.fullText) {
          this.errorMsg = "음성이 감지되지 않았습니다.";
        }
      } catch (e) {
        this.errorMsg = `처리 실패: ${e.message}`;
        this.phase = "idle";
      }
    },
    async _pollUntilDone() {
      const pollStart = Date.now();
      const POLL_TIMEOUT = 20 * 60 * 1000;
      while (Date.now() - pollStart < POLL_TIMEOUT) {
        await new Promise((r) => setTimeout(r, 3000));
        const status = await this.api(`/api/session/${this.sessionId}/status`);
        const diarDone = status.diar_status === "done" || status.diar_status === "error";
        const sttFinalDone = status.stt_final_status === "done" || status.stt_final_status === "error";
        if (status.diar_status === "error") console.warn("diarization error:", status.diar_error);
        if (status.stt_final_status === "error") console.warn("stt final error:", status.stt_final_error);
        if (diarDone && sttFinalDone) break;
      }
    },
    stopStreams() {
      if (this.micStream) {
        this.micStream.getTracks().forEach((t) => t.stop());
        this.micStream = null;
      }
      if (this.sysStream) {
        this.sysStream.getTracks().forEach((t) => t.stop());
        this.sysStream = null;
      }
    },
    cleanup() {
      this.stopStreams();
      if (this.recordingTimer) clearInterval(this.recordingTimer);
      if (this.sendInterval) clearInterval(this.sendInterval);
      if (this.scriptProc) {
        try { this.scriptProc.disconnect(); } catch {}
      }
      if (this.audioCtx && this.audioCtx.state !== "closed") {
        try { this.audioCtx.close(); } catch {}
      }
      this.audioCtx = null;
      this.scriptProc = null;
    },
    // ── File test ─────────────────────────────────────────
    onFileChange(e) {
      this.audioFile = e.target.files && e.target.files[0] ? e.target.files[0] : null;
    },
    async decodeToPcm16(source, targetRate) {
      const buf = await source.arrayBuffer();
      const ctx = new AudioContext();
      const decoded = await ctx.decodeAudioData(buf.slice(0));
      const frameCount = Math.ceil(decoded.duration * targetRate);
      const offline = new OfflineAudioContext(1, frameCount, targetRate);
      const src = offline.createBufferSource();
      src.buffer = decoded;
      src.connect(offline.destination);
      src.start(0);
      const rendered = await offline.startRendering();
      const f32 = rendered.getChannelData(0);
      const pcm = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        pcm[i] = s < 0 ? s * 32768 : s * 32767;
      }
      await ctx.close();
      return pcm;
    },
    async testWithFile() {
      if (!this.audioFile) return;
      this.fileBusy = true;
      this.errorMsg = "";
      try {
        await this.createSession();
        const pcm = await this.decodeToPcm16(this.audioFile, 16000);
        const chunkSamples = 16000 * this.fileChunkSec;
        const totalChunks = Math.ceil(pcm.length / chunkSamples);
        this.phase = "recording";
        this.mode = "file";
        this.recordingTime = 0;
        let chunkErrors = 0;
        const rt = this.realtimeStt ? 1 : 0;
        for (let pos = 0; pos < pcm.length; pos += chunkSamples) {
          const end = Math.min(pcm.length, pos + chunkSamples);
          const part = pcm.subarray(pos, end);
          const bytes = new Uint8Array(part.buffer, part.byteOffset, part.byteLength);
          const seq = this.chunkSeq++;
          try {
            const r = await this.api(`/api/session/${this.sessionId}/chunk?seq=${seq}&realtime=${rt}`, {
              method: "POST",
              headers: { "Content-Type": "application/octet-stream" },
              body: bytes,
            });
            if (this.realtimeStt) {
              this.latestChunkText = r.chunk_text || "";
            }
            this.chunkCount = seq + 1;
            if (r.audio_metrics) this.lastMetrics = r.audio_metrics;
          } catch (ce) {
            chunkErrors++;
            console.error(`chunk ${seq}/${totalChunks} failed:`, ce.message);
            if (ce.status === 413 || ce.status === 429) break;
          }
          this.recordingTime = Math.floor(end / 16000);
        }
        if (chunkErrors > 0) console.warn(`${chunkErrors}/${totalChunks} chunks failed`);
        this.phase = "processing";
        await this.api(`/api/session/${this.sessionId}/stop`, { method: "POST" });
        await this._pollUntilDone();
        const r = await this.api(`/api/session/${this.sessionId}/finalize`, { method: "POST" });
        this.timeline = (r.timeline || []).map((t) => ({ ...t }));
        this.fullText = r.full_text || "";
        this.rawSttItems = r.raw_stt_items || [];
        this.rawDiarSegments = r.raw_diar_segments || [];
        this.rawSttFinalItems = r.raw_stt_final_items || [];
        this.phase = "results";
        if (this.timeline.length === 0 && !this.fullText) {
          this.errorMsg = "음성이 감지되지 않았습니다.";
        }
      } catch (e) {
        console.error("testWithFile error:", e);
        this.errorMsg = `파일 처리 실패: ${e.message}`;
        this.phase = "idle";
      } finally {
        this.fileBusy = false;
      }
    },
    // ── Results / Edit ────────────────────────────────────
    openEdit(idx) {
      this.editIdx = idx;
      this.editSpeaker = this.timeline[idx].speaker;
      this.editText = this.timeline[idx].text;
      this.editOpen = true;
    },
    saveEdit() {
      if (this.editIdx >= 0 && this.editIdx < this.timeline.length) {
        this.timeline[this.editIdx].text = this.editText;
      }
      this.editOpen = false;
    },
    closeEdit() {
      this.editOpen = false;
    },
    resetToIdle() {
      this.phase = "idle";
      this.sessionId = "";
      this.timeline = [];
      this.fullText = "";
      this.latestChunkText = "";
      this.lastMetrics = null;
      this.chunkSeq = 0;
      this.chunkCount = 0;
      this.errorMsg = "";
      this.mode = "";
      this.recordingTime = 0;
      this.showDebug = false;
      this.rawSttItems = [];
      this.rawDiarSegments = [];
      this.rawSttFinalItems = [];
      window.scrollTo(0, 0);
    },
    // ── Helpers ───────────────────────────────────────────
    speakerColor(speaker) {
      const list = this.speakerList;
      const idx = list.indexOf(speaker);
      const shades = ["#0D9488", "#7C3AED", "#D97706", "#2563EB", "#DC2626"];
      return shades[idx % shades.length];
    },
    formatSec(sec) {
      if (sec == null) return "";
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return `${m}:${String(s).padStart(2, "0")}`;
    },
    fmtMs(ms) {
      if (ms == null) return "";
      const totalSec = ms / 1000;
      const m = Math.floor(totalSec / 60);
      const s = (totalSec % 60).toFixed(1);
      return `${m}:${s.padStart(4, "0")}`;
    },
    downloadAudio() {
      if (!this.sessionId) return;
      const url = `${this.apiBase}/api/session/${this.sessionId}/audio`;
      const iframe = document.createElement("iframe");
      iframe.style.display = "none";
      iframe.src = url;
      document.body.appendChild(iframe);
      setTimeout(() => { try { document.body.removeChild(iframe); } catch(e) {} }, 60000);
    },
    downloadResults() {
      const lines = [];
      lines.push("=== 회의 결과 ===");
      lines.push(`세션: ${this.sessionId}`);
      lines.push(`생성: ${new Date().toLocaleString("ko-KR")}`);
      lines.push("");
      for (const t of this.timeline) {
        lines.push(`[${this.formatSec(t.start)} ~ ${this.formatSec(t.end)}] ${t.speaker}`);
        lines.push(t.text);
        lines.push("");
      }
      if (this.fullText) {
        lines.push("=== 전체 텍스트 ===");
        lines.push(this.fullText);
      }
      const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `meeting_${this.sessionId.slice(0, 8)}_${new Date().toISOString().slice(0, 10)}.txt`;
      a.style.display = "none";
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 500);
    },
  },
  beforeUnmount() {
    this.cleanup();
  },
}).mount("#app");
