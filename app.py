import gradio as gr
import os
from dotenv import load_dotenv
from rag_pipeline import chat_with_gemini, extract_text, extract_resume_text, analyze_resume_ats

# Load environment variables
load_dotenv()

import sys
if any("streamlit" in arg for arg in sys.argv) or "streamlit" in sys.modules:
    try:
        import streamlit as st
        st.error("NaanChalant AI has been migrated to Gradio!")
        st.stop()
    except Exception:
        pass

# ── Lucide SVG helper (renders inside gr.HTML only) ──────────────────────────
def icon(name, size=18, color="currentColor", extra=""):
    PATHS = {
        "sparkles":      '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="m5 3 1 2.5L8.5 6 6 7 5 9.5 4 7 1.5 6 4 5.5z"/><path d="m19 17 1 2.5 2.5.5-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1z"/>',
        "upload":        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
        "bar-chart":     '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
        "search":        '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
        "lightbulb":     '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .3 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
        "layout":        '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/>',
        "message":       '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
        "settings":      '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
        "triangle-alert":'<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        "check-circle":  '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        "zap":           '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "star":          '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
        "file-text":     '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>',
        "award":         '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
    }
    path = PATHS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;flex-shrink:0;{extra}">'
        f'{path}</svg>'
    )

def section_header(icon_name, title, subtitle=None, icon_color="#4f46e5"):
    sub = f"<p style='margin:2px 0 0 0;color:#64748b;font-size:0.82rem;font-family:Plus Jakarta Sans,sans-serif;'>{subtitle}</p>" if subtitle else ""
    return (
        f"<div style='display:flex;align-items:center;gap:10px;padding:4px 0 10px 0;'>"
        f"<span style='display:flex;align-items:center;justify-content:center;"
        f"width:36px;height:36px;border-radius:10px;"
        f"background:linear-gradient(135deg,{icon_color}22,{icon_color}11);"
        f"border:1px solid {icon_color}33;flex-shrink:0;'>"
        f"{icon(icon_name, 18, icon_color)}"
        f"</span>"
        f"<div><h3 style='margin:0;font-size:1rem;font-weight:700;"
        f"font-family:Outfit,sans-serif;color:#0f172a;'>{title}</h3>{sub}</div>"
        f"</div>"
    )

# ── Analysis function ─────────────────────────────────────────────────────────
def perform_analysis(resume_file, required_skills):
    warn = lambda msg: (
        f"<div style='display:flex;align-items:center;gap:10px;padding:1.2rem 1.5rem;"
        f"background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);"
        f"border-radius:12px;color:#b91c1c;font-weight:600;font-size:0.9rem;margin:8px 0;'>"
        f"{icon('triangle-alert',18,'#ef4444','margin-right:4px;')}{msg}</div>",
        "<span style='color:#94a3b8;'>—</span>",
        "<span style='color:#94a3b8;'>—</span>",
        "", "", "", "", ""
    )

    if not resume_file:
        return warn("Please upload a resume file first.")
    if not required_skills or required_skills.strip() == "":
        return warn("Please enter the required skills or job description.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() in ("", "YOUR_GEMINI_API_KEY"):
        return warn("API Key is missing. Configure GEMINI_API_KEY in your .env or Space Secrets.")

    resume_text = extract_resume_text(resume_file.name)
    if resume_text.startswith("Error reading"):
        return warn(resume_text)

    analysis = analyze_resume_ats(resume_text, required_skills, api_key=api_key)

    score     = analysis.get("match_score", 0)
    matching  = analysis.get("matching_skills", [])
    missing   = analysis.get("missing_skills", [])
    keywords  = analysis.get("keyword_analysis", "No keyword analysis available.")
    formatting = analysis.get("formatting_issues", [])
    recs      = analysis.get("recommendations", [])

    # Score colour
    score_color = "#ef4444" if score < 50 else ("#f59e0b" if score < 80 else "#10b981")
    score_bg    = "rgba(239,68,68,0.06)" if score < 50 else ("rgba(245,158,11,0.06)" if score < 80 else "rgba(16,185,129,0.06)")
    score_label = "Needs Work" if score < 50 else ("Good Match" if score < 80 else "Excellent Match")
    score_html = f"""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
        padding:24px 16px 16px;'>
      <div style='position:relative;width:150px;height:150px;'>
        <svg viewBox="0 0 36 36" width="150" height="150" style='transform:rotate(-90deg);'>
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="rgba(0,0,0,0.06)" stroke-width="3"/>
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="{score_color}" stroke-width="3"
            stroke-dasharray="{score} {100-score}" stroke-linecap="round"
            style="transition:stroke-dasharray 1s ease;"/>
        </svg>
        <div style='position:absolute;inset:0;display:flex;flex-direction:column;
            align-items:center;justify-content:center;'>
          <span style='font-size:2rem;font-weight:800;font-family:Outfit,sans-serif;
              color:#0f172a;line-height:1;'>{score}%</span>
          <span style='font-size:0.68rem;font-weight:600;color:{score_color};
              text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;'>{score_label}</span>
        </div>
      </div>
      <h3 style='margin:12px 0 4px;font-family:Outfit,sans-serif;font-size:1rem;
          font-weight:700;color:#1e293b;'>ATS Compatibility Score</h3>
      <p style='margin:0;font-size:0.8rem;color:#64748b;'>Based on keyword & skill analysis</p>
    </div>"""

    # Skills pills
    def skill_pill(s, bg, border, color):
        return (f"<span style='display:inline-flex;align-items:center;gap:5px;"
                f"background:{bg};border:1px solid {border};color:{color};"
                f"padding:4px 12px;border-radius:99px;font-size:0.8rem;"
                f"font-weight:600;margin:3px;font-family:Plus Jakarta Sans,sans-serif;'>{s}</span>")

    matching_html = (
        "".join(skill_pill(s, "rgba(16,185,129,0.08)", "rgba(16,185,129,0.25)", "#047857") for s in matching)
        if matching else
        "<span style='color:#94a3b8;font-size:0.88rem;'>No matching skills identified.</span>"
    )
    missing_html = (
        "".join(skill_pill(s, "rgba(239,68,68,0.07)", "rgba(239,68,68,0.2)", "#b91c1c") for s in missing)
        if missing else
        f"<span style='display:inline-flex;align-items:center;gap:6px;color:#047857;"
        f"font-size:0.88rem;font-weight:600;'>{icon('check-circle',16,'#10b981')} No missing skills! Outstanding match.</span>"
    )

    # Markdown lists
    if formatting:
        formatting_md = "\n".join(f"- {item}" for item in formatting)
    else:
        formatting_md = "No significant formatting or structural issues found."

    if recs:
        recs_md = "\n".join(f"- {item}" for item in recs)
    else:
        recs_md = "No immediate changes recommended. The resume is in excellent shape!"

    return (score_html, matching_html, missing_html, keywords, recs_md, formatting_md, resume_text, required_skills)


# ── Chat handler ──────────────────────────────────────────────────────────────
def chat_response(message, history, resume_text, required_skills, model_dropdown, temperature_slider):
    formatted_messages = []
    for msg in history:
        if isinstance(msg, dict):
            role, content = msg.get("role", "user"), msg.get("content", "")
        else:
            role, content = getattr(msg, "role", "user"), getattr(msg, "content", "")
        formatted_messages.append({"role": role, "content": extract_text(content)})

    formatted_messages.append({"role": "user", "content": extract_text(message)})

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() in ("", "YOUR_GEMINI_API_KEY"):
        return "API key is missing. Please configure GEMINI_API_KEY in your .env file or Space Secrets."
    if not resume_text:
        return "Please upload and analyze a resume first — then I can give you tailored suggestions!"
    try:
        return chat_with_gemini(
            formatted_messages, api_key=api_key,
            model_name=model_dropdown, temperature=temperature_slider,
            resume_text=resume_text, required_skills=required_skills
        )
    except Exception as e:
        return f"Error calling Gemini: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  CSS — Complete premium light theme
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root tokens ─────────────────────────────────────────────────────────── */
:root, .gradio-container, .dark, .dark .gradio-container {
    --body-background-fill: #f1f5f9 !important;
    --container-background-fill: rgba(255,255,255,0.75) !important;
    --block-background-fill: rgba(255,255,255,0.8) !important;
    --block-border-color: rgba(99,102,241,0.1) !important;
    --border-color-primary: rgba(99,102,241,0.1) !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: rgba(99,102,241,0.2) !important;
    --input-border-color-focus: #6366f1 !important;
    --input-placeholder-color: #94a3b8 !important;
    --button-primary-background-fill: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    --button-primary-text-color: #ffffff !important;
    --body-text-color: #1e293b !important;
    --body-text-color-subdued: #64748b !important;
    --block-title-text-color: #0f172a !important;
    --slider-color: #6366f1 !important;
    --chatbot-body-background-color: transparent !important;
    --message-user-background-color: linear-gradient(135deg,#4f46e5,#6366f1) !important;
    --message-user-text-color: #ffffff !important;
    --message-bot-background-color: #ffffff !important;
    --message-bot-text-color: #1e293b !important;
    --message-bot-border-color: rgba(99,102,241,0.12) !important;
}

/* ── Global ──────────────────────────────────────────────────────────────── */
body, html, .dark, .dark body {
    background:
        radial-gradient(ellipse at 80% 10%, rgba(99,102,241,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 10% 80%, rgba(139,92,246,0.06) 0%, transparent 50%),
        #f1f5f9 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.gradio-container, .dark .gradio-container {
    background: transparent !important;
}
* { box-sizing: border-box; }

/* ── Typography ──────────────────────────────────────────────────────────── */
h1,h2,h3,h4,h5,h6,p,span,label,
.dark h1,.dark h2,.dark h3,.dark h4,.dark h5,.dark h6,
.dark p,.dark span,.dark label,
.dark .block-title,.dark .block-label {
    color: #1e293b !important;
}

/* ── Blocks / Cards ──────────────────────────────────────────────────────── */
.block, .dark .block {
    background: rgba(255,255,255,0.82) !important;
    border: 1px solid rgba(99,102,241,0.1) !important;
    border-radius: 18px !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.04), 0 1px 0 rgba(255,255,255,0.9) inset !important;
    backdrop-filter: blur(12px) !important;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
#app-header, .dark #app-header {
    background: rgba(255,255,255,0.88) !important;
    backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 20px !important;
    box-shadow: 0 4px 32px rgba(99,102,241,0.08) !important;
    margin-bottom: 20px !important;
    padding: 20px 24px !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
button.primary, .dark button.primary {
    background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 18px rgba(79,70,229,0.35) !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    color: #ffffff !important;
}
button.primary *, .dark button.primary * { color: #ffffff !important; }
button.primary:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(79,70,229,0.45) !important; }
button.primary:active { transform: translateY(0) !important; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.tabs { border-bottom: 1px solid rgba(99,102,241,0.1) !important; }
.tabs button[role="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #64748b !important;
    padding: 9px 16px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s !important;
    border: none !important;
    background: transparent !important;
}
.tabs button[role="tab"]:hover { color: #4f46e5 !important; background: rgba(99,102,241,0.05) !important; }
.tabs button[role="tab"][aria-selected="true"] {
    color: #4f46e5 !important;
    border-bottom: 2px solid #4f46e5 !important;
    background: transparent !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
aside, .sidebar, [class*="sidebar"],
.dark aside, .dark .sidebar, .dark [class*="sidebar"] {
    background: rgba(255,255,255,0.95) !important;
    backdrop-filter: blur(28px) !important;
    border-left: 1px solid rgba(99,102,241,0.15) !important;
    box-shadow: -6px 0 40px rgba(79,70,229,0.08) !important;
}
/* Sidebar open toggle tab */
[class*="sidebar-toggle"], .sidebar-toggle-btn {
    background: linear-gradient(180deg,#4f46e5,#7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px 0 0 12px !important;
    box-shadow: -4px 0 16px rgba(79,70,229,0.35) !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.25s ease !important;
    padding: 14px 8px !important;
}
[class*="sidebar-toggle"]:hover {
    background: linear-gradient(180deg,#4338ca,#6d28d9) !important;
    box-shadow: -4px 0 22px rgba(79,70,229,0.5) !important;
}
[class*="sidebar-toggle"] * { color: #ffffff !important; }

/* ── Chatbot bubble area ─────────────────────────────────────────────────── */
#chatbot-view, .dark #chatbot-view {
    background: linear-gradient(155deg,#f8faff 0%,#eef2ff 100%) !important;
    border: 1px solid rgba(99,102,241,0.14) !important;
    border-radius: 16px !important;
    box-shadow: inset 0 2px 12px rgba(99,102,241,0.06) !important;
}

/* Bot bubbles */
#chatbot-view .message.bot, .dark #chatbot-view .message.bot,
.message.bot, .dark .message.bot {
    background: #ffffff !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 18px 18px 18px 5px !important;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06) !important;
    color: #1e293b !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
}

/* User bubbles */
#chatbot-view .message.user, .dark #chatbot-view .message.user,
.message.user, .dark .message.user {
    background: linear-gradient(135deg,#4f46e5 0%,#6366f1 100%) !important;
    border: none !important;
    border-radius: 18px 18px 5px 18px !important;
    box-shadow: 0 4px 18px rgba(79,70,229,0.28) !important;
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
}
.message.user *, #chatbot-view .message.user *,
.dark .message.user *, .dark #chatbot-view .message.user * {
    color: #ffffff !important;
}

/* ── Chat input row ──────────────────────────────────────────────────────── */
.chat-input-row, [class*="chatinterface"] form,
.dark .chat-input-row, .dark [class*="chatinterface"] form {
    background: #ffffff !important;
    border: 1.5px solid rgba(99,102,241,0.22) !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 14px rgba(99,102,241,0.07) !important;
    padding: 6px 8px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    margin-top: 8px !important;
}
.chat-input-row:focus-within, [class*="chatinterface"] form:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12), 0 2px 14px rgba(99,102,241,0.1) !important;
}
.chat-input-row textarea, [class*="chatinterface"] textarea,
.dark .chat-input-row textarea, .dark [class*="chatinterface"] textarea {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #1e293b !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 6px 8px !important;
}
.chat-input-row textarea::placeholder, [class*="chatinterface"] textarea::placeholder {
    color: #94a3b8 !important;
    font-style: italic !important;
}

/* Send button */
.chat-input-row button[aria-label="Submit"],
[class*="chatinterface"] button[aria-label="Submit"],
[class*="submit"] {
    background: linear-gradient(135deg,#4f46e5,#6366f1) !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 12px rgba(79,70,229,0.3) !important;
    transition: all 0.2s ease !important;
    min-width: 38px !important;
    min-height: 38px !important;
    color: #ffffff !important;
}
.chat-input-row button[aria-label="Submit"]:hover { transform: scale(1.07) !important; }
.chat-input-row button[aria-label="Submit"] svg,
[class*="chatinterface"] button svg { stroke: #ffffff !important; fill: none !important; }

/* ── Accordion ───────────────────────────────────────────────────────────── */
.accordion, .dark .accordion {
    background: rgba(248,250,252,0.9) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    margin-top: 10px !important;
}
.accordion .label-wrap, .dark .accordion .label-wrap {
    background: rgba(99,102,241,0.04) !important;
    padding: 10px 16px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #4f46e5 !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
.accordion .label-wrap:hover { background: rgba(99,102,241,0.09) !important; }

/* ── Inputs ──────────────────────────────────────────────────────────────── */
input, textarea, select, .dark input, .dark textarea, .dark select {
    color: #1e293b !important;
    background: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
input:focus, textarea:focus { box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important; }

/* ── Slider ──────────────────────────────────────────────────────────────── */
input[type=range] { accent-color: #6366f1 !important; }

/* ── File upload area ────────────────────────────────────────────────────── */
.upload-container, [class*="upload"], .file-preview,
.dark .upload-container, .dark [class*="upload"], .dark .file-preview {
    background: rgba(248,250,252,0.8) !important;
    border: 2px dashed rgba(99,102,241,0.25) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
.upload-container:hover, [class*="upload"]:hover {
    border-color: rgba(99,102,241,0.5) !important;
    background: rgba(99,102,241,0.03) !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.22); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.45); }
"""

# ══════════════════════════════════════════════════════════════════════════════
#  UI Layout
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="NaanChalant AI — Resume ATS Analyzer") as demo:
    resume_text_state      = gr.State("")
    required_skills_state  = gr.State("")

    # ── Header ───────────────────────────────────────────────────────────────
    with gr.Row(elem_id="app-header"):
        gr.HTML(
            f"<div style='display:flex;align-items:center;gap:16px;'>"
            f"<div style='display:flex;align-items:center;justify-content:center;"
            f"width:48px;height:48px;border-radius:14px;flex-shrink:0;"
            f"background:linear-gradient(135deg,#4f46e5,#7c3aed);box-shadow:0 4px 16px rgba(79,70,229,0.35);'>"
            f"{icon('sparkles', 24, '#ffffff')}"
            f"</div>"
            f"<div>"
            f"<h1 style='margin:0;font-size:1.65rem;font-weight:800;font-family:Outfit,sans-serif;"
            f"background:linear-gradient(135deg,#4f46e5 10%,#7c3aed 50%,#db2777 90%);"
            f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
            f"NaanChalant AI Resume ATS Analyzer</h1>"
            f"<p style='margin:3px 0 0;color:#64748b;font-size:0.88rem;font-family:Plus Jakarta Sans,sans-serif;'>"
            f"Compare resumes against required skills &amp; get AI-powered rewrite recommendations."
            f"</p>"
            f"</div>"
            f"</div>"
        )

    # ── Main two-column layout ────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left: Inputs
        with gr.Column(scale=1, min_width=320):
            gr.HTML(section_header("upload", "Resume &amp; Requirements",
                                   "Upload your resume and paste the job description"))
            resume_file = gr.File(
                label="Upload Resume (PDF, DOCX, TXT)",
                file_types=[".pdf", ".docx", ".txt"],
                type="filepath"
            )
            required_skills = gr.Textbox(
                label="Required Skills / Job Description",
                placeholder="Paste key skills, qualifications, or the full job description here…",
                lines=11,
                max_lines=24
            )
            analyze_btn = gr.Button(
                f"  Analyze Resume",
                variant="primary",
                elem_id="analyze-btn"
            )

        # Right: Results
        with gr.Column(scale=1, min_width=380):
            gr.HTML(section_header("bar-chart", "ATS Match Evaluation",
                                   "Keyword match score and skill gap analysis"))
            score_widget = gr.HTML(
                value=(
                    "<div style='display:flex;flex-direction:column;align-items:center;"
                    "justify-content:center;padding:3rem 1rem;color:#94a3b8;'>"
                    f"{icon('file-text', 40, '#cbd5e1', 'display:block;margin:0 auto 12px;')}"
                    "<p style='margin:0;font-size:0.95rem;font-family:Plus Jakarta Sans,sans-serif;'>"
                    "Upload a resume and paste job description to run analysis.</p></div>"
                )
            )

            with gr.Tabs(elem_id="result-tabs"):
                with gr.Tab("Skills &amp; Keywords"):
                    gr.HTML(f"<div style='padding:4px 0 6px;'>"
                            f"{icon('check-circle',15,'#10b981','margin-right:5px;')}"
                            f"<strong style='font-size:0.88rem;color:#047857;'>Matching Skills</strong></div>")
                    matching_skills_widget = gr.HTML(
                        value="<span style='color:#94a3b8;font-size:0.88rem;'>Run analysis to see matching skills.</span>"
                    )
                    gr.HTML(f"<div style='padding:10px 0 6px;'>"
                            f"{icon('triangle-alert',15,'#ef4444','margin-right:5px;')}"
                            f"<strong style='font-size:0.88rem;color:#b91c1c;'>Missing Skills</strong></div>")
                    missing_skills_widget = gr.HTML(
                        value="<span style='color:#94a3b8;font-size:0.88rem;'>Run analysis to see skill gaps.</span>"
                    )
                    gr.HTML(f"<div style='padding:10px 0 6px;'>"
                            f"{icon('search',15,'#6366f1','margin-right:5px;')}"
                            f"<strong style='font-size:0.88rem;color:#4f46e5;'>Keyword Analysis</strong></div>")
                    keyword_analysis_widget = gr.Markdown(value="*No keyword analysis available.*")

                with gr.Tab("Recommendations"):
                    gr.HTML(f"<div style='padding:4px 0 10px;'>"
                            f"{icon('zap',15,'#f59e0b','margin-right:5px;')}"
                            f"<strong style='font-size:0.88rem;color:#92400e;'>Actionable Edit Suggestions</strong></div>")
                    recommendations_widget = gr.Markdown(
                        value="*Upload and analyze a resume to see actionable edit checklists.*"
                    )

                with gr.Tab("Formatting"):
                    gr.HTML(f"<div style='padding:4px 0 10px;'>"
                            f"{icon('layout',15,'#6366f1','margin-right:5px;')}"
                            f"<strong style='font-size:0.88rem;color:#3730a3;'>Structural &amp; Formatting Issues</strong></div>")
                    formatting_widget = gr.Markdown(
                        value="*Upload and analyze a resume to detect structural formatting warnings.*"
                    )

    # ── Sidebar Chatbot ───────────────────────────────────────────────────────
    with gr.Sidebar(label="Resume Coach", open=False, position="right"):

        # Sidebar header
        gr.HTML(
            f"<div style='display:flex;align-items:center;gap:10px;"
            f"padding:4px 0 14px;border-bottom:1px solid rgba(99,102,241,0.12);margin-bottom:12px;'>"
            f"<div style='display:flex;align-items:center;justify-content:center;"
            f"width:38px;height:38px;border-radius:10px;flex-shrink:0;"
            f"background:linear-gradient(135deg,#4f46e5,#7c3aed);'>"
            f"{icon('message', 18, '#ffffff')}"
            f"</div>"
            f"<div>"
            f"<h3 style='margin:0;font-size:1rem;font-weight:700;font-family:Outfit,sans-serif;color:#0f172a;'>Resume Coach</h3>"
            f"<p style='margin:0;font-size:0.75rem;color:#64748b;'>AI-powered career advisor</p>"
            f"</div>"
            f"</div>"
        )

        # Quick prompt chips
        gr.HTML(
            "<div style='margin-bottom:12px;'>"
            "<p style='margin:0 0 8px;font-size:0.78rem;font-weight:600;color:#64748b;"
            "text-transform:uppercase;letter-spacing:0.5px;'>Quick Prompts</p>"
            "<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
            "<span style='display:inline-block;padding:5px 10px;border-radius:99px;"
            "border:1px solid rgba(99,102,241,0.25);font-size:0.75rem;font-weight:600;"
            "color:#4f46e5;background:rgba(99,102,241,0.05);cursor:pointer;'>Rewrite summary</span>"
            "<span style='display:inline-block;padding:5px 10px;border-radius:99px;"
            "border:1px solid rgba(99,102,241,0.25);font-size:0.75rem;font-weight:600;"
            "color:#4f46e5;background:rgba(99,102,241,0.05);cursor:pointer;'>Bullet points</span>"
            "<span style='display:inline-block;padding:5px 10px;border-radius:99px;"
            "border:1px solid rgba(99,102,241,0.25);font-size:0.75rem;font-weight:600;"
            "color:#4f46e5;background:rgba(99,102,241,0.05);cursor:pointer;'>Skills gap tips</span>"
            "<span style='display:inline-block;padding:5px 10px;border-radius:99px;"
            "border:1px solid rgba(99,102,241,0.25);font-size:0.75rem;font-weight:600;"
            "color:#4f46e5;background:rgba(99,102,241,0.05);cursor:pointer;'>Cover letter</span>"
            "</div>"
            "</div>"
        )

        chatbot = gr.Chatbot(
            height=380,
            show_label=False,
            elem_id="chatbot-view",
            layout="bubble",
            avatar_images=(None, None),
            placeholder=(
                f"<div style='display:flex;flex-direction:column;align-items:center;"
                f"justify-content:center;padding:2.5rem 1rem;'>"
                f"<div style='width:56px;height:56px;border-radius:16px;margin-bottom:14px;"
                f"background:linear-gradient(135deg,rgba(79,70,229,0.1),rgba(124,58,237,0.1));"
                f"border:1px solid rgba(99,102,241,0.2);display:flex;align-items:center;"
                f"justify-content:center;'>"
                f"{icon('lightbulb', 26, '#6366f1')}"
                f"</div>"
                f"<h4 style='margin:0 0 6px;font-family:Outfit,sans-serif;font-size:0.95rem;"
                f"font-weight:700;color:#1e293b;'>Resume Coach</h4>"
                f"<p style='margin:0;font-size:0.8rem;color:#64748b;text-align:center;'>"
                f"Ask me to rewrite bullet points, generate a professional summary, or fix skill gaps.</p>"
                f"</div>"
            )
        )

        with gr.Accordion("Settings", open=False):
            gr.HTML(
                f"<div style='display:flex;align-items:center;gap:8px;padding:8px 0 4px;'>"
                f"{icon('settings', 15, '#6366f1')}"
                f"<span style='font-size:0.82rem;font-weight:600;color:#4f46e5;'>Model Configuration</span>"
                f"</div>"
            )
            model_dropdown = gr.Dropdown(
                choices=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
                value="gemini-2.5-flash",
                label="Gemini Model",
                interactive=True
            )
            temperature_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                label="Creativity (Temperature)",
                interactive=True
            )

        gr.ChatInterface(
            fn=chat_response,
            chatbot=chatbot,
            additional_inputs=[resume_text_state, required_skills_state, model_dropdown, temperature_slider],
            textbox=gr.Textbox(
                placeholder="Ask me to improve your resume…",
                show_label=False,
                lines=1,
                max_lines=4,
                elem_id="chat-input-box",
                submit_btn="Send"
            )
        )

    # ── Wire events ───────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=perform_analysis,
        inputs=[resume_file, required_skills],
        outputs=[
            score_widget, matching_skills_widget, missing_skills_widget,
            keyword_analysis_widget, recommendations_widget, formatting_widget,
            resume_text_state, required_skills_state
        ]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="violet",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        )
    )