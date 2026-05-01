# AI Tutor Backend

A FastAPI backend that runs a local `mistralai/Mistral-7B-Instruct-v0.2` model with 4-bit quantization and safe, student-oriented answer generation.

## Project Files

- `main.py` - FastAPI app, model loading, prompt engineering, GPU safety, and `/ask` endpoint.
- `requirements.txt` - Python dependency list for FastAPI, Transformers, bitsandbytes, and Torch.

## Installation

1. Create a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

> If you are using a Windows CUDA GPU, install the matching PyTorch wheel from https://pytorch.org/get-started/locally/ before or after this step.

## Run the Server

Start the API with Uvicorn:

```powershell
python main.py
```

The service will be available at `http://127.0.0.1:8000`.

## API Endpoints

### POST /ask

Request body:

```json
{
  "userId": "student123",
  "studentClass": 5,
  "text": "What is photosynthesis?"
}
```

Response body:

```json
{
  "success": true,
  "answer": "Photosynthesis is how plants use sunlight, water, and air to make food."
}
```

### GET /health

Simple health check to verify the server and model state.

### OpenAPI Docs

Visit `http://127.0.0.1:8000/docs` for the automatically generated Swagger UI.

## Notes

- The model is loaded with 4-bit quantization and an async lock to prevent concurrent GPU calls.
- The prompt is tailored for a school teacher persona and uses a student grade level.
- If the model cannot answer safely, it returns `I don't know`.
- Keep question text under 3200 characters to avoid prompt truncation.
