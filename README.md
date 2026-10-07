# ✨ Nexra.ai — AI Tutor

A FastAPI backend plus chat UI that runs an open instruction-tuned LLM (default
`mistralai/Mistral-7B-Instruct-v0.2`, 4-bit quantized) to answer school students'
questions at their class level (Class 1–12).

## Features

- **Live token streaming** over Server-Sent Events, which works through Colab tunnels.
- **Grade-aware teaching prompts**: simple words for young students, step-by-step board-exam-style answers for higher classes.
- **Conversation memory** per student (the last 4 Q&A pairs), so follow-ups like "explain again simpler" work.
- **Proper chat templates** for any model (Mistral, Qwen, Llama, …), including templates that have no system role.
- **Stop button**: generation on the GPU stops as soon as the student presses Stop or closes the tab.
- **Markdown + math rendering** (KaTeX) in the UI, plus a copy button, dark mode, and mobile layout.
- **One GPU, one job**: an async lock keeps parallel requests from running out of memory.
- Works on **T4 (fp16)** and **A100/L4 (bf16)** automatically, and every setting can be changed with environment variables.

## Run on Google Colab (recommended)

1. Push this repo to GitHub, or upload the files in step 2 of the notebook.
2. Open `Nexra_Colab.ipynb` in Colab and choose **Runtime → Change runtime type → T4 GPU**.
3. For gated models (Mistral, Llama): accept the model terms on Hugging Face and add your token as the Colab secret `HF_TOKEN`.
4. Run all cells. The last setup cell prints a public `https://….trycloudflare.com` link. Open it to use the tutor.

## Run locally (needs an NVIDIA GPU with at least 6 GB VRAM)

```bash
pip install -r requirements.txt
python main.py            # http://127.0.0.1:8001
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_ID` | `mistralai/Mistral-7B-Instruct-v0.2` | Any Hugging Face chat model |
| `LOAD_IN_4BIT` | `true` | 4-bit NF4 quantization (about 5 GB VRAM for 7B) |
| `MAX_NEW_TOKENS` | `1024` | Maximum answer length in tokens |
| `MAX_INPUT_TOKENS` | `3072` | Prompt budget; the oldest history is dropped first |
| `TEMPERATURE` / `TOP_P` | `0.3` / `0.9` | Sampling; `TEMPERATURE=0` means greedy |
| `REPETITION_PENALTY` | `1.08` | Reduces loops |
| `HISTORY_TURNS` | `4` | Q&A pairs remembered per user (`0` turns memory off) |
| `PORT` | `8001` | Server port |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI (`index.html`) |
| `POST` | `/ask/stream` | Streams the answer as SSE |
| `POST` | `/ask` | Returns the full answer as JSON |
| `DELETE` | `/history/{userId}` | Clears a student's memory |
| `GET` | `/health` | Model status, GPU, and VRAM |
| `GET` | `/docs` | Swagger UI |

Request body (both ask endpoints):

```json
{ "userId": "student123", "studentClass": 8, "text": "What is photosynthesis?", "useHistory": true }
```

`/ask/stream` events:

```
data: {"token": "Photo"}
data: {"token": "synthesis is ..."}
data: {"done": true, "answer": "...", "stopReason": "stop", "seconds": 6.2}
```

On failure the stream sends `data: {"error": "..."}` instead.

`/ask` response:

```json
{ "success": true, "answer": "...", "stopReason": "stop", "seconds": 6.2 }
```

`stopReason` is `"length"` when the answer hit `MAX_NEW_TOKENS`. The student can then ask "continue".
