# Deepak AI Assistant — Professional Edition

A ChatGPT-style, glassmorphism-themed chat interface built on **Streamlit** + **Groq**.

## File Structure

```
Chatbot/
├── app.py              # Streamlit UI (chat, sidebar, welcome screen, exports)
├── chatbot.py           # Groq client wrapper (streaming, models, error handling)
├── utils.py             # File ingestion, chat export/import, analytics
├── requirements.txt     # Python dependencies
├── .env.example          # Template for local API key
└── .env                  # Your real key (local only, not committed)
```

## What's new vs. the original

| Area | Added |
|---|---|
| UI/UX | Glassmorphism cards, gradient hero title, welcome screen with suggestion cards, dark theme refresh |
| Sidebar | Model selector (6 Groq models), temperature slider, max-token slider, system prompt editor, file upload, conversation search, export (MD/TXT/JSON/PDF), import (JSON), live analytics |
| Chat | Token-by-token **streaming**, native Markdown + code-block rendering, per-message timestamps, copy button, conversation search/filter |
| Files | PDF text extraction, CSV preview, TXT ingestion, image attachment (metadata) — auto-appended to your next message |
| Metrics | Total/user/assistant message counts, estimated tokens, session duration, last response time |
| Errors | Friendly messages for invalid key, rate limits, network failures, empty prompts |
| Code quality | Split into `chatbot.py` / `utils.py` / `app.py`, type hints, docstrings throughout |

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API key (local dev)**
   ```bash
   cp .env.example .env
   # then edit .env and paste your key:
   # GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
   ```
   Get a key at https://console.groq.com/keys

3. **Run locally**
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo (do **not** commit your real `.env` — it's for local use only).
2. Go to https://share.streamlit.io → "New app" → point it at your repo and `app.py`.
3. Since there's no `.env` on Streamlit Cloud, the app will automatically show an API-key input
   in the sidebar — paste your Groq key there. (Alternatively, add `GROQ_API_KEY` under
   **App settings → Secrets** as `GROQ_API_KEY = "gsk_..."` and it will be picked up automatically
   the same way the local `.env` is, since both rely on `os.getenv("GROQ_API_KEY")`.)
4. Click **Deploy**.

## Notes on specific features

- **Image upload**: there's no OCR/vision backend wired in (Groq's text models don't accept
  images), so images are attached as a filename reference only. If you want real image
  understanding, swap in a Groq vision-capable model and pass the image as a multimodal message.
- **Copy button**: Streamlit doesn't support clipboard JS natively without custom components, so
  the 📋 button is a lightweight placeholder — full code blocks already have a built-in copy icon
  from Streamlit's native Markdown/code renderer when you hover over them.
- **PDF export**: built with `fpdf2`; non-Latin-1 characters are safely replaced rather than
  crashing the export.