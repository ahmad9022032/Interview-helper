# Interview Copilot

A fully local, real-time assistant for technical interviews.
It listens to the interviewer through your microphone, transcribes speech as it happens,
detects when a question has been asked, and streams a short, interview-ready answer to
your screen.

```
Click Record → speaker/mic audio → faster-whisper → local LLM (Ollama) → answer on screen
```

You click once when the interviewer starts asking and once when they finish, so a pause in
the middle of a question cannot split it and nothing said outside that window is ever
transcribed.

**Everything runs on your machine.** Audio never leaves your computer: speech-to-text is
faster-whisper running locally, the language model is served by Ollama locally, and the
browser talks to the backend over a local WebSocket. No cloud APIs, no keys, no accounts.

Measured on an Apple M5 laptop, from the moment the interviewer stops speaking:

| | |
|---|---|
| First word of the answer on screen | **≈0.9 s** |
| Complete 2–4 line answer | **≈2.3 s** |

---

## Contents

1. [Requirements](#requirements)
2. [Quick start](#quick-start)
3. [Install Ollama and a model](#1-install-ollama-and-a-model)
4. [Backend](#2-backend-fastapi--faster-whisper)
5. [Frontend](#3-frontend-nextjs)
6. [Configuration](#4-configuration)
7. [Providing your resume / context](#5-providing-your-resume--context)
8. [Using the app](#6-using-the-app)
9. [How the real-time pipeline works](#7-how-the-real-time-pipeline-works)
10. [Latency tuning (with measurements)](#8-latency-tuning)
11. [Choosing a model](#9-choosing-a-model)
12. [Tests](#10-tests)
13. [Docker](#11-docker-optional)
14. [Troubleshooting](#12-troubleshooting)
15. [Project structure](#project-structure)
16. [Privacy](#privacy)

---

## Requirements

- macOS, Linux or Windows (developed and tested on Apple Silicon macOS)
- Python 3.10 – 3.14
- Node.js 18+ and npm
- [Ollama](https://ollama.com)
- ~6 GB free RAM for the recommended model plus Whisper
- A browser with microphone access, on `localhost` or HTTPS

---

## Quick start

```bash
# 1. Ollama + a model
brew install ollama && brew services start ollama     # macOS
ollama pull qwen3.5:4b

# 2. Config
cp .env.example .env

# 3. Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# 4. Frontend (new terminal)
cd frontend && npm install && npm run dev
```

Open <http://localhost:3000>, click **Listen to the interviewer** and share the tab your
interview is in with its audio. Then click **Record question** when the interviewer starts
asking, and click it again when they finish to get the answer.

---

## 1. Install Ollama and a model

### macOS

```bash
brew install ollama
brew services start ollama        # or run `ollama serve` in a terminal
```

Or download the desktop app from <https://ollama.com/download>.

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Download the installer from <https://ollama.com/download>.

### Pull a model

```bash
# Recommended: best answers, same speed as a 3B in practice (~3.4 GB)
ollama pull qwen3.5:4b

# Smaller and shorter, weaker on modern architecture (~1.9 GB)
ollama pull qwen2.5:3b-instruct
```

Verify:

```bash
ollama run qwen3.5:4b "Say hi in three words"
```

Any chat model in Ollama works. See [Choosing a model](#9-choosing-a-model) for measured
numbers and why reasoning models like `qwen3` are a poor fit for live interviews.

---

## 2. Backend (FastAPI + faster-whisper)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs **faster-whisper** (which brings CTranslate2) and **onnxruntime**, used for
the Silero VAD model that ships inside faster-whisper. No ffmpeg and no PyTorch are
needed: audio reaches Whisper as an in-memory NumPy array.

Create your config from the example, in the project root:

```bash
cp .env.example .env
```

Start it:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# or:  python -m app.main
```

On first start faster-whisper downloads the Whisper weights (`small` is ~470 MB) into the
Hugging Face cache. The log ends with:

```
Interview Copilot ready (MODEL=qwen3.5:4b, WHISPER_MODEL=small)
```

Check everything at once:

```bash
curl http://localhost:8000/api/health
```

`status` is `ok` when Whisper is loaded, Ollama is reachable and the configured model is
pulled. Otherwise it is `degraded` and the JSON names the missing piece.

---

## 3. Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev            # http://localhost:3000
```

The frontend reads `NEXT_PUBLIC_BACKEND_URL` (default `http://localhost:8000`). If the
backend runs elsewhere, create `frontend/.env.local`:

```
NEXT_PUBLIC_BACKEND_URL=http://192.168.1.20:8000
```

Browsers only allow microphone access on `localhost` or HTTPS.

---

## 4. Configuration

Everything lives in `.env` in the project root (copy of `.env.example`). Nothing is
hard-coded; the model in particular is read from `MODEL` at startup.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `qwen3.5:4b` | Default model. Can also be switched from the UI dropdown, which takes precedence. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL. |
| `LLM_THINK` | `auto` | Probes the model at startup and configures reasoning correctly. See [below](#reasoning-models). |
| `LLM_THINKING_BUDGET` | `2000` | Extra tokens a reasoning model may spend thinking. |
| `MAX_ANSWER_TOKENS` | `220` | Token budget for the default **Short answer**. The longer settings carry their own budgets. |
| `LLM_TEMPERATURE` | `0.3` | Lower is more stable. |
| `LLM_KEEP_ALIVE` | `30m` | Keeps the model in RAM between questions. |
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`. |
| `WHISPER_DEVICE` | `auto` | `cpu`, `cuda`, `auto`. |
| `WHISPER_COMPUTE_TYPE` | `auto` | `int8` on CPU, `float16` on GPU. |
| `WHISPER_LANGUAGE` | `en` | A fixed language is faster; `auto` detects. |
| `CAPTURE_MODE` | `push` | `push` = the record button marks the question. `auto` = infer it from silence. |
| `PUSH_MAX_S` | `180` | Safety cap if a recording is somehow left running. |
| `PUSH_MIN_S` | `0.4` | Recordings shorter than this are ignored. |
| `VAD_THRESHOLD` | `0.5` | Speech probability threshold. Raise it in a noisy room. Used by `auto`, and to trim silence in `push`. |
| `VAD_SILENCE_MS` | `700` | Silence that ends a question. |
| `VAD_MIN_SPEECH_MS` | `300` | Shorter bursts are treated as noise. |
| `PARTIAL_INTERVAL_S` | `1.2` | How often the live transcript refreshes while someone is speaking. |
| `MIN_QUESTION_WORDS` | `4` | Shorter fragments wait to be merged with the next one. |
| `QUESTION_HOLD_S` | `1.5` | How long a fragment waits before being answered anyway. |
| `HISTORY_TURNS` | `3` | Previous Q&A pairs kept for follow-up questions. |
| `MAX_CONTEXT_CHARS` | `6000` | Cap on the resume/profile text put in the prompt. |
| `BACKEND_PORT` / `FRONTEND_PORT` | `8000` / `3000` | Ports. |

---

## 5. Providing your resume / context

The assistant answers project and behavioral questions from **your** background, and is
instructed never to invent experience that is not in the context. Two sources combine:

1. **Structured profile** — edit [`config/candidate_profile.yaml`](config/candidate_profile.yaml):
   name, summary, skills, projects (name / description / technologies / outcome), work
   experience, education, preferred technologies. It is re-read on the next answer; you
   can also `POST /api/context/reload`.
2. **Free-form resume text** — in the UI open **Resume / candidate context**, paste your
   resume and press *Save*, or upload a **PDF** (text extracted with pypdf). It is stored
   locally at `data/resume.txt`.

Both are rendered into one `CANDIDATE CONTEXT` block in the system prompt, truncated at
`MAX_CONTEXT_CHARS` to keep prompts short and answers fast.

Grounding works: asked about experience that is not in the profile, the assistant says so
rather than inventing it.

---

## 6. Using the app

1. Start Ollama, the backend and the frontend.
2. Open <http://localhost:3000>.
3. Pick a **model** from the header dropdown if you want to trade speed for depth, and a
   **mode**, which changes the system prompt:
   - **Technical** — programming, ML, AI, system design, computer science
   - **Behavioral** — strengths, weaknesses, motivation, teamwork
   - **Project** — questions about your own resume and projects
4. Click **Listen to the interviewer**. The browser asks what to share: pick the tab your
   interview is in and tick **Also share tab audio**.
5. **Click Record question when the interviewer starts asking, and click it again when they
   finish.** The button turns red and counts up while it is recording. The transcript
   appears, the question is shown, and the answer streams in.
6. **Stop** pauses capture. **Clear** wipes the transcript, the answer and the rolling
   memory used for follow-ups.

### How much answer you get

The dropdown next to the capture buttons sets the length. It applies from the next question
and is remembered between sessions.

| Setting | Length | Use it for |
|---|---|---|
| **Short answer** (default) | 2-5 lines | Almost everything. Read it straight out. |
| **Detailed** | 6-8 lines | When one line would lose the trade-off that matters. |
| **Step by step** | ~10 numbered lines | "Walk me through how you would..." |
| **Architecture explanation** | 8-14 lines | "Explain the architecture of X", naming each component and the data flow. |

Each setting carries its own token budget, so a longer answer is genuinely longer rather than
a short one that runs out mid-sentence. Measured on the same question, short produced 69
words and architecture produced 186.

### Why you mark the question yourself

Recording marks exactly what the question is, and that removes a whole class of errors.
Without it the app has to infer where a question ended from silence, and it gets that wrong
whenever the interviewer pauses to think mid-sentence:

> "Can you explain what a transformer is" ... *(0.8s pause)* ... "and why we use attention?"

Guessing from silence splits that into two questions, throws away the first, and answers the
meaningless fragment. There is no way to tell a thinking pause from a finished question by
looking at the words. "Can you explain what a transformer is" is a perfectly complete
sentence, and Whisper even puts a question mark on it. Only you know the interviewer had
not finished.

While you are not recording, nothing is transcribed and nothing is sent to the model, so small
talk, your own answers and the interviewer reading your resume aloud are all ignored.

- Click slightly late and you still keep the first 1.5 seconds, which is buffered continuously.
- Stop slightly late and the trailing silence is trimmed before transcription.
- A stray click with no speech in it is rejected, not sent to the model.
- If the connection drops or a capture runs past `PUSH_MAX_S`, the button resets itself rather
  than sitting there claiming to still be recording.

**Automatic** mode is still available next to the capture setting, and behaves the way it did
before: silence of `VAD_SILENCE_MS` is treated as the end of a question. It needs no clicking,
but a mid-question pause longer than that will split the question.

Follow-up questions work: ask *"What is RAG?"* then *"Why would you use it instead of
fine-tuning?"* and the second answer knows that "it" means RAG.

You can also type a question in the **Ask** box to test without a microphone.

The latency panel shows speech-to-text time, time to the model's first token, total model
time, time from end of speech to the first answer word, and the total.

### Where the audio comes from

**Listen to the interviewer** (the default) captures the audio coming *out of your speakers*
rather than going into your microphone. This matters for two reasons:

- Your own voice is never captured, so the app never transcribes your answer and tries to
  answer it back. Only the interviewer triggers a response.
- Nothing has to be audible in the room, and no microphone permission is involved.

Browsers expose this through screen sharing, so you choose a source when you start:

| Interview runs in | Pick |
|---|---|
| Google Meet, Zoom Web, Teams Web, any browser tab | that **tab**, with *Also share tab audio* ticked |
| Zoom / Teams desktop app, Windows | **Entire Screen**, with *Share system audio* ticked |
| Zoom / Teams desktop app, macOS | see the loopback note below |

macOS does not let Chrome capture whole-system audio, only tab audio. If your interview is
in a desktop app rather than a browser tab, route the system output through a virtual device
and use the **Use microphone instead** button with it selected:

- Install [BlackHole](https://github.com/ExistentialAudio/BlackHole), create a *Multi-Output
  Device* (your speakers + BlackHole) in Audio MIDI Setup, set it as the system output, then
  choose *BlackHole* as the microphone in the browser's site settings.

**Use microphone instead** is the fallback everywhere else. It picks up the interviewer
through your speakers, with echo cancellation, noise suppression and auto gain all disabled
so the browser does not filter their voice away. The trade-off is that it also hears you, so
your own answers may be transcribed and answered.

### Does it start on its own?

Browsers require a click before any audio capture, for obvious privacy reasons, so the first
start of a session is always manual. After that:

- A **microphone** session resumes by itself on reload, because the permission is remembered.
  Reopening the page mid-interview puts you straight back to listening with no click.
- A **speaker capture** session always needs one click, because browsers deliberately never
  let a page re-attach to your screen or tab audio silently. This is not something the app
  can work around.

---

## 7. How the real-time pipeline works

Nothing is recorded to disk and nothing waits for the interview to end.

1. **Microphone → PCM chunks.** An `AudioWorklet` resamples the mic from the device rate
   (44.1/48 kHz) to 16 kHz mono, converts to 16-bit PCM and posts 1024-sample chunks
   (~64 ms). Each is sent as a binary WebSocket frame.
2. **Voice activity detection.** The backend runs Silero VAD — the ONNX model bundled with
   faster-whisper, driven statefully one 32 ms frame at a time — over every chunk. Speech
   opens an utterance buffer with ~320 ms of pre-roll so the first syllable is not clipped.
   `VAD_SILENCE_MS` of silence closes it.
3. **Incremental transcription.** While the interviewer is still speaking, the buffer is
   transcribed every `PARTIAL_INTERVAL_S` and shown as a grey partial line. Partials are
   never started once a pause begins, because a transcription running in a worker thread
   cannot be cancelled and would delay the final one — the one the answer waits on.
4. **Final transcription.** On end of speech the buffer, with trailing silence trimmed, goes
   through faster-whisper once more (greedy, fixed language). One model instance is loaded at
   startup and reused; work runs in a thread so the event loop keeps accepting audio.
5. **Where the question starts and ends.** In the default `push` mode you say so: audio is
   buffered only while the record button is active, plus 1.5 s of pre-roll from before you
   clicked, with silence trimmed off both ends before transcription. Nothing else is
   transcribed at all.
   In `auto` mode the text is instead accepted as a question if it ends in `?`, starts with
   interview phrasing ("what", "why", "tell me", …), or is at least `MIN_QUESTION_WORDS` long,
   with short fragments held and merged into the next utterance.
6. **LLM.** The question, the mode's system prompt, your candidate context and the last
   `HISTORY_TURNS` Q&A pairs (answers truncated) go to Ollama's `/api/chat` with
   `stream: true` and `keep_alive`, over a pooled HTTP connection. Tokens stream to the UI as
   they arrive, and a new question cancels an answer still in flight.
7. **Events.** The client receives `status`, `transcript_partial`, `transcript_final`,
   `question`, `answer_delta`, `answer_reset`, `answer_done` (with latency numbers) and
   `error`.

Failures are contained. A failed transcription sends *"Could not understand audio. Please
repeat."*, an unreachable Ollama sends *"AI model unavailable. Check Ollama."*, a denied
microphone shows *"Microphone access required."* — and the session keeps listening.

---

## 8. Latency tuning

Measured on an Apple M5, CPU int8, 8.3 s of speech over three clips:

| `WHISPER_MODEL` | per utterance | speed | word error rate |
|---|---|---|---|
| `tiny` | 111 ms | 0.04× realtime | 2.8% |
| `base` | 230 ms | 0.08× realtime | 0% |
| `small` (default) | 783 ms | 0.28× realtime | 0% |

**Setting `WHISPER_MODEL=base` removes about 550 ms from every answer** and was just as
accurate on clean speech. The default is `small` because it is markedly more robust with
accents, background noise and poor microphones, which is what a real interview sounds like.
Try `base` first; move back to `small` if you see transcription mistakes.

Other knobs, in rough order of effect:

- **Smaller LLM** — see [Choosing a model](#9-choosing-a-model).
- **`VAD_SILENCE_MS=500`** reacts faster, at the risk of cutting a question at a mid-sentence
  pause. `900` is safer for slow speakers.
- **`MAX_ANSWER_TOKENS=150`** for shorter answers that finish sooner.
- **GPU**: with NVIDIA set `WHISPER_DEVICE=cuda` and `WHISPER_COMPUTE_TYPE=float16`.
- Keep `LLM_KEEP_ALIVE` generous and `LLM_WARMUP=true` so the model is resident before the
  first question. The first answer after startup is always the slowest.

Measure it yourself:

```bash
cd backend && source .venv/bin/activate
python tests/bench_stt.py --models tiny base small
python tests/bench_llm.py --models qwen2.5:3b-instruct qwen2.5:7b-instruct
```

---

## 9. Choosing a model

Switch models from the **dropdown in the header** at any time. The change applies to the
next question with no restart, and is remembered in `data/selected_model.txt`, which
overrides `MODEL` in `.env`. Only models you have already pulled are listed.

### Measured on this project

Eight "explain the architecture of X" questions covering RAG, LangChain, LangGraph, agents,
vector databases and transformers. *Concepts* is the share of the key ideas a good interview
answer should mention, scored automatically, so it rewards substance over fluency.
Apple M5, `temperature=0`:

| `MODEL` | concepts | 1st word | full answer | length |
|---|---|---|---|---|
| **qwen3.5:4b** (default) | **84%** | 556 ms | 3.0 s | 71 words |
| qwen3.5:9b | 82% | 966 ms | 5.1 s | 72 words |
| qwen2.5:7b-instruct | 66% | 160 ms | 5.2 s | 106 words |
| qwen2.5:3b-instruct | 62% | 84 ms | 1.2 s | 52 words |
| qwen3:4b (always-on reasoning) | — | 13.6 s | 14.8 s | 42 words |

`qwen3.5:4b` is the recommendation: it is far better on modern architecture than the older
Qwen2.5 models, whose training predates most of what interviewers now ask about. The 9B is
no better and twice as slow, so there is nothing to gain by going bigger.

**In the real pipeline that quality is free.** End to end, the first word of the answer
reaches the screen 864-902 ms after the interviewer stops speaking, which is what
`qwen2.5:3b-instruct` did too. Speech-to-text (~800 ms) dominates, and the model's first
token only costs 60-97 ms once it is warm. The full answer takes about a second longer only
because it says more, and since the text streams you are already reading it.

Reproduce any of this yourself:

```bash
cd backend && source .venv/bin/activate
python tests/bench_quality.py --models qwen3.5:4b qwen2.5:3b-instruct   # substance
python tests/bench_llm.py     --models qwen3.5:4b qwen2.5:3b-instruct   # latency
```

A note on `qwen3.5:4b-mlx`: the Apple-Silicon build measures faster in isolation (195 ms to
first token) but was consistently *slower* inside the real pipeline on this machine
(~500 ms), where speech-to-text runs on the CPU between requests. Benchmark it yourself
before switching.

### Reasoning models

Thinking-capable models split into two camps, and the backend probes once at startup to tell
them apart, because the difference is worth seconds per answer:

- **Hybrids** such as `qwen3.5` genuinely stop reasoning when asked, and then answer
  immediately. These are left in fast mode.
- **Always-on reasoners** such as `qwen3` and `deepseek-r1` ignore the switch: they write
  their chain-of-thought into the answer itself and spend the whole token budget on it,
  often producing no answer at all. For these the backend turns thinking *on*, because
  Ollama then returns the reasoning in a separate field and the answer stays clean. It also
  raises the budget by `LLM_THINKING_BUDGET` and warns you in the UI.

The probe is one short generation: a model that honours the switch answers "why is the sky
blue" in a sentence and stops, while one that ignores it runs to the token cap. Set
`LLM_THINK` to `true` or `false` to override.

Either way the answer stays clean. The streaming parser also strips inline `<think>` blocks,
and if a stray closing tag reveals that what it streamed was reasoning, it tells the UI to
discard it and start again.

## 10. Tests

With the backend running:

```bash
cd backend && source .venv/bin/activate

python tests/test_units.py          # chain-of-thought stripping, question detection, history
python tests/test_errors.py         # Ollama down, missing model, silence, bad PDF
python tests/test_push_to_ask.py    # click-to-record: pauses, discarded audio, stray clicks
python tests/test_answer_depth.py   # each answer-length setting produces that much answer
python tests/test_pause_bug.py      # the bug push mode fixes (run it with CAPTURE_MODE=auto)
python tests/test_ws_pipeline.py    # full audio pipeline over the WebSocket
python tests/test_conversation.py   # follow-ups, modes, grounding, clear
python tests/test_context_api.py    # health, resume paste, PDF upload
```

`test_ws_pipeline.py` synthesizes a spoken question with macOS `say` and streams it to the
backend at real-time pace, exactly as the browser does, then prints the latency breakdown.
It also accepts your own audio or a typed question:

```bash
python tests/test_ws_pipeline.py --wav my_question.wav
python tests/test_ws_pipeline.py --ask "Why would you use RAG instead of fine-tuning?"
```

---

## 11. Docker (optional)

Ollama stays on the host, where the hardware acceleration is. The backend and frontend run
in containers and reach it through `host.docker.internal`.

```bash
cp .env.example .env
docker compose up --build
```

Whisper weights are cached in the `whisper-models` volume. The microphone is captured by the
browser, so no audio devices need to be passed into Docker. Whisper runs on CPU with `int8`
in the container; running natively is faster.

---

## 12. Troubleshooting

**"Microphone access required."**
The browser denied or could not find a microphone. Click the padlock in the address bar and
allow it, then press Start again. This only works on `localhost` or HTTPS. On macOS also
check *System Settings → Privacy & Security → Microphone*.

**"That source has no audio."**
You shared a window or a screen instead of a tab, or left the audio checkbox unticked. Share
a **browser tab** and tick *Also share tab audio*. On macOS, Chrome cannot capture
whole-screen audio at all, only tab audio.

**It answers my own voice, not just the interviewer's.**
You are on the microphone fallback, which hears everything in the room. Use **Listen to the
interviewer** instead: capturing speaker output means your voice is never in the stream.

**Only my voice is picked up, not the interviewer's.**
On the microphone fallback, the browser or OS is cancelling speaker audio. Interview Copilot
already disables echo cancellation, noise suppression and auto gain. If it still happens, use
speaker capture, a loopback device (BlackHole / Stereo Mix), raise the speaker volume, or
lower `VAD_THRESHOLD` to about `0.35`.

**"Could not understand audio. Please repeat."**
Whisper returned nothing for an utterance longer than a second, usually background noise or
a very quiet signal. Watch the input level bar while someone speaks; raise `VAD_THRESHOLD`
in a noisy room, or move from `base` to `small`.

**The transcript shows "Thank you." when nobody is talking.**
A classic Whisper hallucination on near-silence. The common ones are filtered out. If you
see others, raise `VAD_THRESHOLD` or `VAD_MIN_SPEECH_MS`.

**Questions are cut in half, or answered too early.**
This is what `push` mode exists to prevent: record the whole question and a pause cannot split
it. If you are deliberately using `auto` mode, raise `VAD_SILENCE_MS` to `900` so a mid-sentence
pause does not end the question, or lower it to `500` if it reacts too slowly.

**The record button is missing or greyed out.**
It only appears once capture is running, so start with **Listen to the interviewer** or **Use
microphone instead** first — the bar should say Ready, not Idle. It also greys out while the
previous question is still being transcribed or answered, and it is hidden entirely in
**Automatic** mode, which needs no clicking.

**"Too short. Record the whole question before stopping."**
The recording was stopped almost immediately, or covered only silence. Record for the duration
of the question. `PUSH_MIN_S` sets the threshold.

**"AI model unavailable. Check Ollama."**

```bash
ollama serve                 # or: brew services start ollama
ollama list                  # is MODEL from .env in the list?
ollama pull qwen3.5:4b
curl http://localhost:8000/api/health
```

**Answers take 10 seconds or more.**
You are probably on an always-on reasoning model — the UI shows a warning banner when you
are. Pick `qwen3.5:4b` from the header dropdown. Otherwise the first request after startup
is always slower because the model is being loaded; keep `LLM_WARMUP=true`.

**"The model returned an empty answer."**
A reasoning model spent its whole budget thinking. Raise `LLM_THINKING_BUDGET`, or switch to
a non-reasoning model.

**An answer stops mid-sentence.**
It hit `MAX_ANSWER_TOKENS`; the UI says so under the answer. Raise it in `.env`.

**The backend cannot load Whisper.**
Use `int8` or `auto` for `WHISPER_COMPUTE_TYPE` on CPU — `float16` is GPU-only. The first run
needs internet to fetch the weights from Hugging Face; `HF_HOME` changes the cache location.
`/api/health` reports the exact error under `whisper.error`.

**`pip install` fails building ctranslate2 or onnxruntime.**
Use a 64-bit Python 3.10–3.14. On Apple Silicon use the arm64 Python from Homebrew or
python.org, not an x86 build under Rosetta.

**The frontend says "backend offline".**
Start the backend and check `NEXT_PUBLIC_BACKEND_URL`. An `https://` page cannot open a
`ws://` socket, so use `http://localhost:3000` in development.

---

## Project structure

```
.
├── .env.example                   all configuration (copy to .env)
├── docker-compose.yml
├── config/
│   └── candidate_profile.yaml     your skills, projects, experience, education
├── data/
│   └── resume.txt                 pasted or PDF-extracted resume (created by the UI)
├── backend/
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── tests/                     unit, error, pipeline, conversation, API tests + benchmarks
│   └── app/
│       ├── main.py                FastAPI app, model loading, /ws endpoint
│       ├── config.py              settings from .env
│       ├── pipeline.py            per-connection VAD → STT → question → LLM pipeline
│       ├── question_detector.py   end-of-question heuristics and fragment merging
│       ├── history.py             rolling conversation memory
│       ├── context_store.py       profile YAML + resume text/PDF
│       ├── stt/vad.py             streaming Silero VAD (onnxruntime)
│       ├── stt/whisper.py         faster-whisper engine
│       ├── llm/ollama_client.py   streaming Ollama client, capability detection
│       ├── prompts/               system prompt, modes, message builder
│       └── routers/context.py     /api/health, /api/context, /api/context/pdf
└── frontend/
    ├── package.json, next.config.js, tsconfig.json, Dockerfile
    ├── public/audio-worklet.js    mic → 16 kHz Int16 PCM chunks
    └── src/
        ├── app/                   layout, page, styles, icon
        ├── components/            Header, Controls, Transcript, Answer, Latency, History, ContextPanel
        └── lib/                   useAudioCapture, useCopilotSocket, types
```

## Privacy

- Microphone audio is streamed only to the backend on your own machine, and is never
  written to disk.
- Speech-to-text runs locally with faster-whisper. The language model runs locally with
  Ollama. No audio, transcript or answer is sent to any external service.
- Your profile and resume are plain files in `config/` and `data/`, and are only ever sent
  to the local model. `data/resume.txt` is in `.gitignore`, so it is not committed if you
  share this repository.
- The only network access the app ever needs is the one-time download of the Whisper
  weights and the Ollama model.
