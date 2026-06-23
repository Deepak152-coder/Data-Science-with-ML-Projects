"""
utils.py
========
Helper utilities for the Deepak AI Assistant:
- File ingestion (PDF, CSV, TXT, image OCR-free preview)
- Chat export (Markdown, TXT, JSON, PDF)
- Chat import
- Token / message analytics
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PyPDF2 import PdfReader
from fpdf import FPDF


# =========================
# FILE INGESTION
# =========================

def read_txt(file) -> str:
    """Read a plain text upload and return decoded text."""
    raw = file.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def read_pdf(file) -> str:
    """Extract text content from an uploaded PDF file."""
    reader = PdfReader(file)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def read_csv_preview(file, max_rows: int = 20) -> str:
    """Return a markdown preview (and shape info) of an uploaded CSV."""
    df = pd.read_csv(file)
    shape_info = f"CSV shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
    preview = df.head(max_rows).to_markdown(index=False)
    return shape_info + preview


def process_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """
    Dispatch an uploaded file to the correct reader based on extension.

    Returns:
        (kind, extracted_text_or_description)
    """
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        text = read_pdf(uploaded_file)
        return "pdf", text[:8000]  # cap to keep prompt reasonable

    if name.endswith(".csv"):
        text = read_csv_preview(uploaded_file)
        return "csv", text[:8000]

    if name.endswith(".txt"):
        text = read_txt(uploaded_file)
        return "txt", text[:8000]

    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        # No OCR backend wired up; we attach the image as context metadata only.
        return "image", f"[Image attached: {uploaded_file.name}]"

    return "unknown", f"[Unsupported file type: {uploaded_file.name}]"


# =========================
# EXPORTS
# =========================

def _visible_messages(messages: List[Dict]) -> List[Dict]:
    return [m for m in messages if m.get("role") != "system"]


def export_as_markdown(messages: List[Dict]) -> str:
    """Render the conversation as a Markdown string."""
    lines = ["# Deepak AI Assistant — Conversation Export", ""]
    lines.append(f"_Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    for msg in _visible_messages(messages):
        role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        ts = msg.get("timestamp", "")
        lines.append(f"### {role} {f'({ts})' if ts else ''}")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def export_as_txt(messages: List[Dict]) -> str:
    """Render the conversation as plain text."""
    lines = []
    for msg in _visible_messages(messages):
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
        lines.append("")
    return "\n".join(lines)


def export_as_json(messages: List[Dict]) -> str:
    """Render the full conversation (including system prompt) as JSON."""
    return json.dumps(messages, indent=2, ensure_ascii=False)


def export_as_pdf(messages: List[Dict]) -> bytes:
    """Render the conversation as a downloadable PDF (bytes)."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=15, top=15, right=15)
    pdf.add_page()
    effective_width = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(effective_width, 10, "Deepak AI Assistant - Conversation Export", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(effective_width, 8, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for msg in _visible_messages(messages):
        role = "You" if msg["role"] == "user" else "Assistant"
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(effective_width, 6, role)
        pdf.set_font("Helvetica", "", 10)
        # Encode safely for FPDF's default font (latin-1); drop unsupported chars.
        safe_text = msg["content"].encode("latin-1", errors="replace").decode("latin-1")
        safe_text = safe_text if safe_text.strip() else "(empty)"
        pdf.multi_cell(effective_width, 6, safe_text)
        pdf.ln(2)

    return bytes(pdf.output(dest="S"))


def import_chat_from_json(uploaded_file) -> Optional[List[Dict]]:
    """
    Parse an uploaded JSON chat export back into the messages list.

    Returns:
        list[dict] | None: parsed messages, or None if invalid.
    """
    try:
        raw = uploaded_file.read()
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, list) and all("role" in m and "content" in m for m in data):
            return data
        return None
    except Exception:
        return None


# =========================
# ANALYTICS
# =========================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (~4 chars per token), good enough for UI display.
    Avoids pulling in a heavy tokenizer dependency.
    """
    return max(1, len(text) // 4)


def compute_analytics(messages: List[Dict], session_start: datetime) -> Dict:
    """Compute session-level analytics for the sidebar."""
    visible = _visible_messages(messages)
    user_msgs = [m for m in visible if m["role"] == "user"]
    assistant_msgs = [m for m in visible if m["role"] == "assistant"]

    total_tokens = sum(estimate_tokens(m["content"]) for m in visible)
    duration = datetime.now() - session_start

    return {
        "total_messages": len(visible),
        "user_messages": len(user_msgs),
        "assistant_messages": len(assistant_msgs),
        "estimated_tokens": total_tokens,
        "session_duration": str(duration).split(".")[0],  # H:MM:SS
    }