import asyncio
import logging
from typing import Optional
from transformers import TextIteratorStreamer
from threading import Thread
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_INPUT_CHARS = 4200
MAX_NEW_TOKENS = 1000
MAX_OUTPUT_CHARS = 2500
MAX_STUDENT_CLASS = 12
MIN_STUDENT_CLASS = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_tutor")

app = FastAPI(
    title="AI Tutor Backend",
    description="A FastAPI backend that runs a local Mistral 7B Instruct model to answer student questions safely.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development (later restrict this)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generate_lock = asyncio.Lock()
model = None
tokenizer = None
generation_config = None
model_device: Optional[torch.device] = None


class AskRequest(BaseModel):
    userId: str = Field(..., min_length=1, description="Unique student or session identifier")
    studentClass: int = Field(
        ..., ge=MIN_STUDENT_CLASS, le=MAX_STUDENT_CLASS, description="Student grade level from 1 to 12"
    )
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS, description="Question text for the tutor")


class AskResponse(BaseModel):
    success: bool
    answer: str


def get_model_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer() -> AutoTokenizer:
    tokenizer_obj = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer_obj.pad_token is None:
        tokenizer_obj.pad_token = tokenizer_obj.eos_token
    return tokenizer_obj


def load_model() -> AutoModelForCausalLM:
    device = get_model_device()
    logger.info("Loading model %s on %s", MODEL_ID, device)

    if device.type == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        try:
            model_obj = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                quantization_config=quant_config,
                device_map="auto",
                max_memory={"0": "8GB"},
            )
            return model_obj
        except Exception as ex:
            logger.warning("4-bit automatic device mapping failed: %s", ex)
            logger.info("Retrying with explicit cuda device map and reduced settings.")
            try:
                model_obj = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                    quantization_config=quant_config,
                    device_map={"": 0},
                )
                return model_obj
            except Exception as second_ex:
                logger.error("Fallback 4-bit load also failed: %s", second_ex)
                raise RuntimeError(
                    "Unable to load the model in 4-bit mode. Confirm CUDA is available and bitsandbytes is installed correctly."
                ) from second_ex

    logger.warning("CUDA is unavailable. Falling back to CPU model load (not 4-bit).")
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="cpu",
    )


def build_prompt(student_class: int, question: str) -> str:
    grade_text = f"Grade {student_class}"
    return (
        "You are a kind and precise school teacher. "
        "Answer in a simple way that a student can understand. "
        "Do not provide inappropriate or irrelevant content. "
        "Always gives full answer."
        "If you are not sure of the answer, say 'I don't know'.\n\n"
        f"Student level: {grade_text}\n"
        "Question: "
        + question.strip()
        + "\n\nAnswer:"
    )


def trim_answer(answer: str) -> str:
    cleaned = answer.strip()
    if len(cleaned) > MAX_OUTPUT_CHARS:
        cleaned = cleaned[: MAX_OUTPUT_CHARS - 3].rstrip() + "..."
    return cleaned


@app.on_event("startup")
async def startup_event():
    global tokenizer, model, generation_config, model_device
    model_device = get_model_device()
    logger.info("Selected device for model inference: %s", model_device)
    tokenizer = load_tokenizer()
    model = load_model()

    generation_config = GenerationConfig(
        temperature=0.25,
        top_p=0.92,
        repetition_penalty=1.05,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        # early_stopping=True,
    )

    logger.info("Model loaded and ready. GPU available: %s", torch.cuda.is_available())

# @app.post("/ask/stream")
# async def ask(request: AskRequest):
#     if len(request.text.strip()) == 0:
#         raise HTTPException(status_code=400, detail="Question text must not be empty.")

#     prompt = build_prompt(request.studentClass, request.text)

#     async def generate_stream():
#         async with generate_lock:
#             try:
#                 inputs = tokenizer(
#                     prompt,
#                     return_tensors="pt",
#                     truncation=True,
#                     max_length=MAX_INPUT_CHARS,
#                     padding="longest",
#                 )

#                 input_ids = inputs.input_ids.to(model_device)
#                 attention_mask = inputs.attention_mask.to(model_device)

#                 # Generate full output (same as before)
#                 with torch.no_grad():
#                     output = model.generate(
#                         input_ids=input_ids,
#                         attention_mask=attention_mask,
#                         generation_config=generation_config,
#                         max_new_tokens=MAX_NEW_TOKENS,
#                         pad_token_id=tokenizer.eos_token_id,
#                         eos_token_id=tokenizer.eos_token_id,
#                     )

#                 raw_answer = tokenizer.decode(
#                     output[0][input_ids.shape[-1]:],
#                     skip_special_tokens=True
#                 )

#                 answer = trim_answer(raw_answer)

#                 if not answer:
#                     answer = "I don't know"

#                 # 🔥 STREAM TOKEN BY TOKEN (fake streaming but smooth UI)
#                 words = answer.split(" ")

#                 for word in words:
#                     yield word + " "
#                     await asyncio.sleep(0.02)  # controls speed

#             except Exception as e:
#                 logger.exception("Streaming failed")
#                 yield "Error generating response."

#     return StreamingResponse(generate_stream(), media_type="text/plain")

@app.post("/ask/stream")
async def ask(request: AskRequest):
    if len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question text must not be empty.")

    prompt = build_prompt(request.studentClass, request.text)

    def generate_stream():
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_CHARS,
                padding="longest",
            )

            input_ids = inputs.input_ids.to(model_device)
            attention_mask = inputs.attention_mask.to(model_device)

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                streamer=streamer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.25,
                top_p=0.92,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            full_text = ""

            for new_text in streamer:
                full_text += new_text
                yield new_text

            # ✅ Final safety: ensure complete sentence
            cleaned = full_text.strip()

            if not cleaned.endswith((".", "!", "?", ":")):
                cleaned += "."

        except Exception:
            logger.exception("Streaming failed")
            yield "Error generating response."

    return StreamingResponse(generate_stream(), media_type="text/plain")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "model": MODEL_ID, "gpu": torch.cuda.is_available()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")
