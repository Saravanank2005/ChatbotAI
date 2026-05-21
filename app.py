import gradio as gr
import os
from dotenv import load_dotenv
from rag_pipeline import chat_with_gemini, extract_text, extract_resume_text, analyze_resume_ats

# Load environment variables
load_dotenv()

# SVG Icons for a clean, premium visual design
SVG_ICONS = {
    "sparkles": """<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; color: #6366f1; margin-right: 6px;"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="m5 3 1 2.5L8.5 6 6 7 5 9.5 4 7 1.5 6 4 5.5z"/><path d="m19 17 1 2.5 2.5.5-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1z"/></svg>""",
    "upload": """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 8px; color: #4f46e5;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>""",
    "chart": """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 8px; color: #4f46e5;"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>""",
    "search": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 6px; color: #6366f1;"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>""",
    "recommendations": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 6px; color: #6366f1;"><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>""",
    "bulb": """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: block; margin: 0 auto 10px auto; color: #eab308;"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A5 5 0 0 0 8 8c0 1 .3 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>""",
    "formatting": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 6px; color: #6366f1;"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>""",
    "chat": """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 8px; color: #4f46e5;"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>""",
    "warning": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 6px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>""",
    "warning_inline": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 4px; margin-top: -2px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>""",
    "success": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 6px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>""",
    "success_inline": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 4px; margin-top: -2px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>""",
    "tip_inline": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 4px; margin-top: -2px;"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>""",
}

import sys
# Guard against running with Streamlit
if any("streamlit" in arg for arg in sys.argv) or "streamlit" in sys.modules:
    try:
        import streamlit as st
        st.error("**NaanChalant AI has been migrated to Gradio!**")
        st.info("Please do not run this script with `streamlit run app.py`.")
        st.markdown(
            "To launch the new Gradio interface:\n"
            "1. Stop/kill this streamlit process (e.g., press `Ctrl+C` in your terminal).\n"
            "2. Run the Gradio app using Python directly:\n"
            "```powershell\n"
            ".\\venv\\Scripts\\python app.py\n"
            "```"
        )
        st.stop()
    except Exception:
        pass

def perform_analysis(resume_file, required_skills):
    if not resume_file:
        return (
            f"<div style='text-align: center; color: #f87171; font-weight: 600; padding: 1.5rem;'>{SVG_ICONS['warning']} Please upload a resume file first.</div>",
            "<span style='color: #64748b;'>No data</span>",
            "<span style='color: #64748b;'>No data</span>",
            "Please upload a resume first.",
            "Please upload a resume first.",
            "Please upload a resume first.",
            "",
            ""
        )
    if not required_skills or required_skills.strip() == "":
        return (
            f"<div style='text-align: center; color: #f87171; font-weight: 600; padding: 1.5rem;'>{SVG_ICONS['warning']} Please enter the required skills or job description.</div>",
            "<span style='color: #64748b;'>No data</span>",
            "<span style='color: #64748b;'>No data</span>",
            "Please enter job description first.",
            "Please enter job description first.",
            "Please enter job description first.",
            "",
            ""
        )
        
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() in ("", "YOUR_GEMINI_API_KEY"):
        return (
            f"<div style='text-align: center; color: #f87171; font-weight: 600; padding: 1.5rem;'>{SVG_ICONS['warning']} API Key is missing. Configure GEMINI_API_KEY in your .env file or Space Secrets.</div>",
            "<span style='color: #64748b;'>No data</span>",
            "<span style='color: #64748b;'>No data</span>",
            "API key missing.",
            "API key missing.",
            "API key missing.",
            "",
            ""
        )
        
    # Extract resume text
    resume_text = extract_resume_text(resume_file.name)
    if resume_text.startswith("Error reading"):
        return (
            f"<div style='text-align: center; color: #f87171; font-weight: 600; padding: 1.5rem;'>{SVG_ICONS['warning']} {resume_text}</div>",
            "<span style='color: #64748b;'>No data</span>",
            "<span style='color: #64748b;'>No data</span>",
            resume_text,
            resume_text,
            resume_text,
            "",
            ""
        )
        
    # Perform analysis
    analysis = analyze_resume_ats(resume_text, required_skills, api_key=api_key)
    
    score = analysis.get("match_score", 0)
    matching = analysis.get("matching_skills", [])
    missing = analysis.get("missing_skills", [])
    keywords = analysis.get("keyword_analysis", "No keyword analysis available.")
    formatting = analysis.get("formatting_issues", [])
    recs = analysis.get("recommendations", [])
    
    # Generate visual HTML score
    score_color = "#ef4444"
    if score >= 80:
        score_color = "#10b981"
    elif score >= 50:
        score_color = "#f59e0b"
        
    score_html = f"""
    <div style='text-align: center; margin: 15px 0;'>
        <div style='display: inline-block; width: 140px; height: 140px; border-radius: 50%; border: 10px solid rgba(0,0,0,0.05); border-top-color: {score_color}; position: relative; box-shadow: 0 10px 25px rgba(99, 102, 241, 0.1);'>
            <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 2.2rem; font-weight: 800; font-family: "Outfit", sans-serif; color: #0f172a;'>{score}%</div>
        </div>
        <h3 style='margin: 10px 0 0 0; font-family: "Outfit", sans-serif; font-size: 1.2rem; color: #1e293b;'>ATS Compatibility Score</h3>
    </div>
    """
    
    # Matching skills tags HTML
    if matching:
        matching_html = "".join([f"<span style='display: inline-block; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); color: #047857; padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; margin: 4px; font-weight: 600;'>{s}</span>" for s in matching])
    else:
        matching_html = "<span style='color: #64748b; font-size: 0.9rem;'>No matching skills identified.</span>"
        
    # Missing skills tags HTML
    if missing:
        missing_html = "".join([f"<span style='display: inline-block; background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); color: #b91c1c; padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; margin: 4px; font-weight: 600;'>{s}</span>" for s in missing])
    else:
        missing_html = f"<span style='color: #047857; font-size: 0.9rem; font-weight: 600;'>{SVG_ICONS['success_inline']} No missing skills! Outstanding match.</span>"
        
    # Formatting bullet lists
    if formatting:
        formatting_md = "\n".join([f"- {SVG_ICONS['warning_inline']} {item}" for item in formatting])
    else:
        formatting_md = f"{SVG_ICONS['success_inline']} No significant formatting or structural issues found."
        
    # Recommendations bullet lists
    if recs:
        recs_md = "\n".join([f"- {SVG_ICONS['tip_inline']} {item}" for item in recs])
    else:
        recs_md = f"{SVG_ICONS['success_inline']} No immediate changes recommended. The resume is in excellent shape!"
        
    return (
        score_html,
        matching_html,
        missing_html,
        keywords,
        recs_md,
        formatting_md,
        resume_text,
        required_skills
    )

def chat_response(message, history, resume_text, required_skills, model_dropdown, temperature_slider):
    """
    Callback function for Gradio ChatInterface.
    history is a list of ChatMessage objects.
    """
    formatted_messages = []
    
    # Map previous conversation history
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
        
        content_str = extract_text(content)
        formatted_messages.append({"role": role, "content": content_str})
        
    # Append the new message
    message_str = extract_text(message)
    formatted_messages.append({"role": "user", "content": message_str})
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() in ("", "YOUR_GEMINI_API_KEY"):
        return "Error: Gemini API key is missing. Please configure GEMINI_API_KEY in your .env file or Hugging Face Space secrets."
    
    if not resume_text:
        return "Please upload and analyze a resume first so that I can provide specific suggestions tailored to your background!"
        
    try:
        response = chat_with_gemini(
            formatted_messages, 
            api_key=api_key, 
            model_name=model_dropdown, 
            temperature=temperature_slider,
            resume_text=resume_text,
            required_skills=required_skills
        )
        return response
    except Exception as e:
        return f"Error calling Gemini: {e}"

# Custom styling with Light Theme and modern colors (forced light mode)
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Style root variables for Gradio elements - Forced Light Theme */
:root, .gradio-container, .dark, .dark .gradio-container {
    --body-background-fill: radial-gradient(circle at top right, rgba(99, 102, 241, 0.05), transparent 45%),
                            radial-gradient(circle at bottom left, rgba(168, 85, 247, 0.05), transparent 45%),
                            #f8fafc !important;
    --container-background-fill: rgba(255, 255, 255, 0.75) !important;
    --block-background-fill: rgba(255, 255, 255, 0.6) !important;
    --block-border-color: rgba(0, 0, 0, 0.06) !important;
    --block-border-width: 1px !important;
    --border-color-primary: rgba(0, 0, 0, 0.06) !important;
    --border-color-secondary: rgba(0, 0, 0, 0.04) !important;
    
    --input-background-fill: #ffffff !important;
    --input-border-color: rgba(0, 0, 0, 0.1) !important;
    --input-border-color-focus: #6366f1 !important;
    --input-placeholder-color: #94a3b8 !important;
    
    --button-primary-background-fill: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
    --button-primary-text-color: #ffffff !important;
    
    --button-secondary-background-fill: rgba(241, 245, 249, 0.8) !important;
    --button-secondary-background-fill-hover: rgba(226, 232, 240, 0.9) !important;
    --button-secondary-text-color: #334155 !important;
    
    /* Text Color Tokens */
    --body-text-color: #1e293b !important;
    --body-text-color-subdued: #64748b !important;
    --block-title-text-color: #0f172a !important;
    --block-label-text-color: #334155 !important;
    --slider-color: #6366f1 !important;

    /* Chatbot Bubble Colors */
    --chatbot-body-background-color: transparent !important;
    --message-user-background-color: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    --message-user-text-color: #ffffff !important;
    --message-user-border-color: rgba(99, 102, 241, 0.2) !important;
    --message-user-border-width: 1px !important;
    
    --message-bot-background-color: #ffffff !important;
    --message-bot-text-color: #1e293b !important;
    --message-bot-border-color: rgba(0, 0, 0, 0.06) !important;
    --message-bot-border-width: 1px !important;
}

/* Force light backgrounds globally */
body, html, .dark, .dark body, .dark html {
    background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.05), transparent 45%),
                radial-gradient(circle at bottom left, rgba(168, 85, 247, 0.05), transparent 45%),
                #f8fafc !important;
}

.gradio-container, .dark .gradio-container {
    background: transparent !important;
}

/* Ensure text contrast on light mode */
h1, h2, h3, h4, h5, h6, p, span, label,
.dark h1, .dark h2, .dark h3, .dark h4, .dark h5, .dark h6, .dark p, .dark span, .dark label,
.dark .block-title, .dark .block-label {
    color: #1e293b !important;
}

/* Header title gradient */
#header-container h1, .dark #header-container h1 {
    background: linear-gradient(135deg, #4f46e5 10%, #7c3aed 50%, #db2777 90%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

/* Header container styling */
#header-container, .dark #header-container {
    margin-bottom: 1.5rem !important;
    padding: 1.2rem 1.5rem !important;
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03) !important;
}

/* Maintain white text for primary buttons */
button.primary, button.primary *, 
.dark button.primary, .dark button.primary * {
    color: #ffffff !important;
}

/* Maintain white text inside user message bubbles */
.message.user, .message.user *, .dark .message.user, .dark .message.user * {
    color: #ffffff !important;
}

/* Adjust primary buttons */
button.primary {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.2) !important;
}
button.primary:active {
    transform: translateY(0) !important;
}
"""

with gr.Blocks(title="NaanChalant AI Resume ATS Analyzer", css=custom_css) as demo:
    # State variables to hold parsed resume and required skills
    resume_text_state = gr.State("")
    required_skills_state = gr.State("")

    # Sleek low-profile header row
    with gr.Row(elem_id="header-container"):
        gr.HTML(
            f"<div style='display: flex; align-items: center; gap: 14px;'>"
            f"{SVG_ICONS['sparkles']}"
            f"<div>"
            f"<h1 style='margin: 0; font-size: 1.8rem; font-weight: 800; font-family: \"Outfit\", sans-serif; background: linear-gradient(135deg, #818cf8 10%, #a78bfa 50%, #f472b6 90%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>NaanChalant AI Resume ATS Analyzer</h1>"
            f"<p style='margin: 2px 0 0 0; color: #94a3b8; font-size: 0.9rem; font-family: \"Plus Jakarta Sans\", sans-serif;'>Compare resumes against required skills with advanced metrics & get AI-powered rewrite recommendations.</p>"
            f"</div>"
            f"</div>"
        )

    with gr.Row():
        # Left Panel (Inputs)
        with gr.Column(scale=1):
            gr.Markdown(f"### {SVG_ICONS['upload']} Resume & Requirements")
            resume_file = gr.File(
                label="Upload Resume (PDF, DOCX, TXT)",
                file_types=[".pdf", ".docx", ".txt"],
                type="filepath"
            )
            required_skills = gr.Textbox(
                label="Required Skills / Job Description",
                placeholder="Paste key skills, qualifications, or the full job description here...",
                lines=12,
                max_lines=25
            )
            analyze_btn = gr.Button("Analyze Resume", variant="primary")

        # Right Panel (Results)
        with gr.Column(scale=1.2):
            gr.Markdown(f"### {SVG_ICONS['chart']} ATS Match Evaluation")
            score_widget = gr.HTML(
                value="<div style='text-align: center; color: #64748b; padding: 3rem;'><p style='font-size: 1.1rem;'>Upload a resume and paste job description to run ATS analysis.</p></div>"
            )
            
            with gr.Tabs():
                with gr.Tab("Skills & Keywords"):
                    gr.Markdown(f"#### {SVG_ICONS['success_inline']} Matching Skills (Found)")
                    matching_skills_widget = gr.HTML(value="<span style='color: #64748b; font-size: 0.95rem;'>No analysis performed yet.</span>")
                    gr.Markdown(f"#### {SVG_ICONS['warning_inline']} Missing Skills (Gaps)")
                    missing_skills_widget = gr.HTML(value="<span style='color: #64748b; font-size: 0.95rem;'>No analysis performed yet.</span>")
                    gr.Markdown(f"#### {SVG_ICONS['search']} Keyword Analysis")
                    keyword_analysis_widget = gr.Markdown(value="*No keyword analysis available.*")
                    
                with gr.Tab("Recommendations"):
                    gr.Markdown(f"#### {SVG_ICONS['recommendations']} Actionable Edit Suggestions")
                    recommendations_widget = gr.Markdown(value="*Upload and analyze a resume to see actionable edit checklists.*")
                    
                with gr.Tab("Formatting & Readability"):
                    gr.Markdown(f"#### {SVG_ICONS['formatting']} Structural & Formatting Issues")
                    formatting_widget = gr.Markdown(value="*Upload and analyze a resume to detect structural formatting warnings.*")

    # Collapsible Corner Sidebar Chatbot
    with gr.Sidebar(label="Resume Coach", open=False, position="right"):
        gr.Markdown(f"### {SVG_ICONS['chat']} Resume Coach")
        gr.Markdown(
            "I'm fully aware of your uploaded resume and the target job description. Ask me questions like:\n"
            "- *'How do I rewrite my project to include Java?'*\n"
            "- *'Write bullet points for React using metric values.'*\n"
            "- *'Generate a professional summary for this JD.'*"
        )
        
        chatbot = gr.Chatbot(
            height=460,
            show_label=False,
            elem_id="chatbot-view",
            layout="bubble",
            placeholder=f"<div style='text-align: center; padding-top: 2rem; color: #64748b;'>{SVG_ICONS['bulb']}<h4>Ask optimization questions here!</h4></div>"
        )
        
        # Accordion for advanced model config inside sidebar
        with gr.Accordion("Settings", open=False):
            model_dropdown = gr.Dropdown(
                choices=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
                value="gemini-2.5-flash",
                label="Select Gemini Model",
                interactive=True
            )
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.05,
                label="Creativity (Temperature)",
                interactive=True
            )
            
        gr.ChatInterface(
            fn=chat_response,
            chatbot=chatbot,
            additional_inputs=[resume_text_state, required_skills_state, model_dropdown, temperature_slider]
        )

    # Wire up the analyze button
    analyze_btn.click(
        fn=perform_analysis,
        inputs=[resume_file, required_skills],
        outputs=[
            score_widget,
            matching_skills_widget,
            missing_skills_widget,
            keyword_analysis_widget,
            recommendations_widget,
            formatting_widget,
            resume_text_state,
            required_skills_state
        ]
    )

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Soft()
    )