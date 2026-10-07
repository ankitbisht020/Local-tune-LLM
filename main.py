"""Nexra.ai - AI Tutor backend.

FastAPI server that runs an instruction-tuned LLM (default: Mistral 7B Instruct)
with 4-bit quantization, real token streaming (Server-Sent Events), per-student
conversation memory and grade-aware teaching prompts.

Every setting can be overridden with an environment variable, e.g.
    MODEL_ID=Qwen/Qwen2.5-7B-Instruct python main.py
"""

import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Deque, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
LOAD_IN_4BIT = _env_bool("LOAD_IN_4BIT", True)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "3072"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.08"))
STREAM_TIMEOUT_SECONDS = float(os.getenv("STREAM_TIMEOUT_SECONDS", "120"))

HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))  # question/answer pairs remembered per user
MAX_TRACKED_USERS = int(os.getenv("MAX_TRACKED_USERS", "1000"))

MIN_STUDENT_CLASS = 1
MAX_STUDENT_CLASS = 12
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ai_tutor")


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
class ModelState:
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    device: Optional[torch.device] = None
    supports_system_role: bool = True
    ready: bool = False
    load_error: Optional[str] = None


state = ModelState()
# One GPU, one generation at a time: prevents out-of-memory under concurrent requests.
generate_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Conversation memory (in-process, bounded, LRU over users)
# ---------------------------------------------------------------------------
class ConversationStore:
    def __init__(self, max_users: int, max_turns: int) -> None:
        self._max_users = max_users
        self._max_messages = max_turns * 2
        self._data: "OrderedDict[str, Deque[Dict[str, str]]]" = OrderedDict()

    def get(self, user_id: str) -> List[Dict[str, str]]:
        history = self._data.get(user_id)
        if history is None:
            return []
        self._data.move_to_end(user_id)
        return list(history)

    def add_turn(self, user_id: str, question: str, answer: str) -> None:
        if self._max_messages <= 0:
            return
        history = self._data.setdefault(user_id, deque(maxlen=self._max_messages))
        self._data.move_to_end(user_id)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        while len(self._data) > self._max_users:
            self._data.popitem(last=False)

    def clear(self, user_id: str) -> None:
        self._data.pop(user_id, None)


conversations = ConversationStore(MAX_TRACKED_USERS, HISTORY_TURNS)


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    userId: str = Field(..., min_length=1, max_length=128, description="Unique student or session identifier")
    studentClass: int = Field(
        ..., ge=MIN_STUDENT_CLASS, le=MAX_STUDENT_CLASS, description="Student grade level from 1 to 12"
    )
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS, description="Question text for the tutor")
    useHistory: bool = Field(True, description="Include this user's recent conversation as context")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question text must not be empty.")
        return value


class AskResponse(BaseModel):
    success: bool
    answer: str
    stopReason: str
    seconds: float


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(device: torch.device) -> AutoModelForCausalLM:
    logger.info("Loading model %s on %s (4-bit=%s)", MODEL_ID, device, LOAD_IN_4BIT and device.type == "cuda")

    if device.type != "cuda":
        logger.warning("CUDA is unavailable: loading on CPU in float32. Expect very slow answers.")
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cpu")

    # T4 (free Colab) has no bfloat16 support; A100/L4 do.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {"torch_dtype": compute_dtype, "device_map": "auto", "low_cpu_mem_usage": True}
    if LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)


def detect_system_role_support(tokenizer: AutoTokenizer) -> bool:
    """Some chat templates (e.g. Mistral v0.1/v0.2) reject a 'system' message."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


def load_everything() -> None:
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state.tokenizer = load_tokenizer()
    state.model = load_model(state.device)
    state.model.eval()
    state.supports_system_role = detect_system_role_support(state.tokenizer)
    logger.info("System role supported by chat template: %s", state.supports_system_role)

    # Warm-up so the first real student does not pay the CUDA kernel start-up cost.
    with torch.inference_mode():
        warm = state.tokenizer("Hello", return_tensors="pt").to(state.model.device)
        state.model.generate(**warm, max_new_tokens=2, do_sample=False, pad_token_id=state.tokenizer.pad_token_id)

    if state.device.type == "cuda":
        used = torch.cuda.memory_allocated() / 1024**3
        logger.info("Model ready on %s, %.1f GB VRAM allocated", torch.cuda.get_device_name(0), used)
    state.ready = True


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
def grade_guidance(student_class: int) -> str:
    if student_class <= 2:
        return (
            "The student is very young. Use very short sentences and very simple words. "
            "Explain with fun, everyday examples (toys, food, animals, family). Keep it to a few lines."
        )
    if student_class <= 5:
        return (
            "Use simple words and short sentences. Explain with familiar real-life examples. "
            "Break ideas into small numbered steps when helpful."
        )
    if student_class <= 8:
        return (
            "Explain clearly with a simple definition first, then an example. "
            "Introduce correct subject terms but explain each one. Use bullet points for lists."
        )
    if student_class <= 10:
        return (
            "Answer the way a good board-exam teacher would: a precise definition, the key points, "
            "formulas or laws where relevant, and a worked example for numericals."
        )
    return (
        "Give a thorough, accurate senior-secondary level answer: precise definitions, derivations or "
        "reasoning where relevant, formulas in LaTeX ($...$), and a step-by-step worked solution for numericals."
    )


def system_prompt(student_class: int) -> str:
    return (
        "You are Nexra, a kind, patient and highly accurate school teacher. "
        f"You are teaching a student of Class {student_class}.\n\n"
        f"How to answer: {grade_guidance(student_class)}\n\n"
        "Rules:\n"
        "- Answer the question fully and finish your explanation; never stop mid-sentence.\n"
        "- Use Markdown formatting (headings, bullet points, **bold** key terms) when it improves clarity.\n"
        "- For maths and science problems, show every step and state the final answer clearly.\n"
        "- Be factually correct. If you are not sure, say \"I don't know\" instead of guessing.\n"
        "- Keep all content safe and appropriate for school students. Politely refuse harmful, adult "
        "or unrelated requests and steer back to learning.\n"
        "- If the student only greets you, greet them back warmly and ask what they want to learn."
    )


def build_messages(request: AskRequest) -> List[Dict[str, str]]:
    history = conversations.get(request.userId) if request.useHistory else []
    sys_text = system_prompt(request.studentClass)

    if state.supports_system_role:
        return [{"role": "system", "content": sys_text}, *history, {"role": "user", "content": request.text}]

    # Template has no system role: put instructions into the first user turn.
    messages = [*history, {"role": "user", "content": request.text}]
    first = dict(messages[0])
    first["content"] = f"{sys_text}\n\n---\n\n{first['content']}"
    messages[0] = first
    return messages


def encode_prompt(request: AskRequest) -> Dict[str, torch.Tensor]:
    """Tokenize with the model's chat template, dropping the oldest history if it is too long."""
    tokenizer = state.tokenizer
    messages = build_messages(request)

    while True:
        # Render to text first: the tensor return type of apply_chat_template differs across versions.
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
        if input_ids.shape[-1] <= MAX_INPUT_TOKENS:
            break
        if len(messages) > 3 and messages[0]["role"] == "system":
            del messages[1:3]  # drop the oldest question/answer pair, keep the system prompt
        elif len(messages) > 2 and messages[0]["role"] != "system":
            # instructions live in messages[0]; move them onto the next user turn when dropping it
            sys_prefix = messages[0]["content"].split("\n\n---\n\n", 1)[0]
            del messages[0:2]
            messages[0] = {**messages[0], "content": f"{sys_prefix}\n\n---\n\n{messages[0]['content']}"}
        else:
            input_ids = input_ids[:, -MAX_INPUT_TOKENS:]  # last resort: keep the newest tokens
            break

    input_ids = input_ids.to(state.model.device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
class StopOnEvent(StoppingCriteria):
    """Stops generation as soon as the client disconnects or presses Stop."""

    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


def _next_chunk(streamer: TextIteratorStreamer) -> Optional[str]:
    try:
        return next(streamer)
    except StopIteration:
        return None


async def stream_answer(ask: AskRequest, http_request: Request) -> AsyncIterator[Dict]:
    """Yields {"token": str} events, then one {"done": True, ...} or {"error": str} event."""
    if not state.ready:
        yield {"error": state.load_error or "Model is still loading. Please try again in a moment."}
        return

    final: Dict
    async with generate_lock:
        started = time.perf_counter()
        inputs = encode_prompt(ask)
        prompt_tokens = inputs["input_ids"].shape[-1]
        stop_event = threading.Event()
        errors: List[BaseException] = []
        generated: List[int] = []
        streamer = TextIteratorStreamer(
            state.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=STREAM_TIMEOUT_SECONDS
        )
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=TEMPERATURE > 0,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=state.tokenizer.pad_token_id,
            eos_token_id=state.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnEvent(stop_event)]),
        )
        if TEMPERATURE > 0:
            generation_kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)

        def run_generation() -> None:
            try:
                with torch.inference_mode():
                    output = state.model.generate(**generation_kwargs)
                generated.append(output.shape[-1] - prompt_tokens)
            except BaseException as exc:  # surface the error to the consumer instead of hanging
                errors.append(exc)
                streamer.end()

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        parts: List[str] = []
        disconnected = False
        try:
            while True:
                chunk = await asyncio.to_thread(_next_chunk, streamer)
                if chunk is None:
                    break
                if await http_request.is_disconnected():
                    disconnected = True
                    break
                if chunk:
                    parts.append(chunk)
                    yield {"token": chunk}
        except Exception as exc:
            logger.exception("Streaming failed")
            errors.append(exc)
        finally:
            stop_event.set()
            await asyncio.to_thread(thread.join)
            if state.device is not None and state.device.type == "cuda":
                torch.cuda.empty_cache()

        seconds = time.perf_counter() - started
        if disconnected:
            logger.info("Client %s disconnected; generation stopped", ask.userId)
            return
        if errors:
            logger.error("Generation error for %s: %r", ask.userId, errors[0])
            final = {"error": "The tutor hit an error while answering. Please try again."}
        else:
            answer = "".join(parts).strip() or "I don't know."
            answer_tokens = generated[0] if generated else 0
            stop_reason = "length" if answer_tokens >= MAX_NEW_TOKENS else "stop"
            conversations.add_turn(ask.userId, ask.text, answer)
            logger.info(
                "Answered %s: %d prompt tokens, %d answer tokens, %.1fs (%.1f tok/s)",
                ask.userId, prompt_tokens, answer_tokens, seconds, answer_tokens / max(seconds, 1e-6),
            )
            final = {"done": True, "answer": answer, "stopReason": stop_reason, "seconds": round(seconds, 2)}

    # Yielded after the lock is released so the next student is never blocked by a slow consumer.
    yield final


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await asyncio.to_thread(load_everything)
    except Exception as exc:
        state.load_error = f"Model failed to load: {exc}"
        logger.exception("Model failed to load")
    yield


app = FastAPI(
    title="Nexra.ai - AI Tutor Backend",
    description="FastAPI backend that runs an instruction-tuned LLM to answer student questions safely.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def index():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found next to main.py")
    return FileResponse(INDEX_HTML)


@app.post("/ask/stream")
async def ask_stream(ask: AskRequest, request: Request):
    """Streams the answer as Server-Sent Events: `data: {"token": ...}` ... `data: {"done": true, ...}`."""

    async def sse() -> AsyncIterator[str]:
        async for event in stream_answer(ask, request):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        # Disable proxy buffering so tokens arrive live through ngrok / cloudflared tunnels.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/ask", response_model=AskResponse)
async def ask(ask: AskRequest, request: Request) -> AskResponse:
    """Non-streaming variant: waits for the full answer."""
    async for event in stream_answer(ask, request):
        if "error" in event:
            raise HTTPException(status_code=503, detail=event["error"])
        if event.get("done"):
            return AskResponse(
                success=True, answer=event["answer"], stopReason=event["stopReason"], seconds=event["seconds"]
            )
    raise HTTPException(status_code=499, detail="Client disconnected.")


@app.delete("/history/{user_id}")
async def clear_history(user_id: str) -> dict:
    conversations.clear(user_id)
    return {"success": True}


@app.get("/health")
async def health_check() -> dict:
    info = {
        "status": "ok" if state.ready else ("error" if state.load_error else "loading"),
        "ready": state.ready,
        "model": MODEL_ID,
        "gpu": torch.cuda.is_available(),
        "busy": generate_lock.locked(),
    }
    if state.load_error:
        info["error"] = state.load_error
    if torch.cuda.is_available():
        info["gpuName"] = torch.cuda.get_device_name(0)
        info["vramAllocatedGB"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
    return info


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
