# AI Interview Preparation Assistant

A private, session-scoped interview preparation workspace built with Streamlit, LangChain, OpenRouter, Sentence Transformers, FAISS, and PyPDF. Upload a resume and study PDFs, generate role-aware questions, evaluate draft answers, and run a question-by-question mock interview.

## Requirements

- Python 3.10 through 3.14
- An OpenRouter API key

The default model is OpenRouter's `openrouter/free` router. To pin a specific model, set `OPENROUTER_MODEL` to a model ID ending in `:free`. The app rejects other model IDs and never falls back to a paid model. Free model availability, limits, and latency are controlled by OpenRouter and can change.

## Local setup

From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add your key to `.env` as `OPENROUTER_API_KEY=...`. Do not commit `.env`. The optional `OPENROUTER_MODEL` setting accepts only `openrouter/free` or an explicitly free model ID ending in `:free`.

Run the app:

```bash
streamlit run app.py
```

Run the tests:

```bash
pytest -q
```

The first document indexing run downloads the small `sentence-transformers/all-MiniLM-L6-v2` embedding model. It runs on CPU; Streamlit caches it for the current process. PDFs and the FAISS index remain in the current Streamlit session's memory and are discarded when the session ends or the app restarts.

## Streamlit Community Cloud

1. Push this project to a GitHub repository. Keep `.env` out of the repository.
2. In Streamlit Community Cloud, create an app from that repository, set the main file to `app.py`, and select Python 3.12 in the advanced settings.
3. In the app's **Settings → Secrets**, add:

   ```toml
   OPENROUTER_API_KEY = "your-openrouter-key"
   # Optional; only `openrouter/free` or a model ID ending in `:free` is accepted.
   OPENROUTER_MODEL = "openrouter/free"
   ```

4. Deploy. Dependencies are installed from `requirements.txt`, including CPU-only PyTorch for the local embedding model.

Documents are processed in memory, with no persistent upload or vector-index directory. The small CPU embedding model is initialized only when documents are indexed and cached within the running app process. Cloud restarts clear the session data and model cache; users can upload their documents again. Free-tier model rate limits may require retrying later.

## Privacy and grounding

- API keys are read from local environment/`.env` or Streamlit Secrets; they are not displayed or logged by the app.
- PDF bytes are parsed in memory. The app does not save uploads to disk or retain them in a persistent index.
- When an AI action is requested, relevant document excerpts, the job description, and/or the submitted answer are sent to OpenRouter to generate that response. The app does not persist these materials to disk.
- Retrieved source labels come from PDF page metadata and are shown separately from generated text. Model output is instructed to use supplied evidence and not claim unsupported candidate facts; verify generated advice before using it.
- Clearing the session removes the app's document index and interview state. For a key accidentally shared outside its intended secret store, revoke it and create a replacement.