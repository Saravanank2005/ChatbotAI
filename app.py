import gradio as gr
import os
from dotenv import load_dotenv
from rag_pipeline import chat_with_gemini, extract_text, extract_resume_text, analyze_resume_ats

# Load environment variables
load_dotenv()

import sys
# Guard against running with Streamlit
if any("streamlit" in arg for arg in sys.argv) or "streamlit" in sys.modules:
    try:
        import streamlit as st
        st.error("⚠️ **NaanChalant AI has been migrated to Gradio!**")
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
            "<div style='text-align: center; color: #f87171; font-weight: 600; padding: 1.5rem;'>⚠️ Please upload a resume file first.</div>",
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
            "<div style='text-align: center; color: #f87171; font-weight: 600; padding: 1.5rem;'>⚠️ Please enter the required skills or job description.</div>",
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
            "<div style='text-align: center; color: #f87171; font-weight: 600; padding: 1.5rem;'>⚠️ API Key is missing. Configure GEMINI_API_KEY in your .env file or Space Secrets.</div>",
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
            f"<div style='text-align: center; color: #f87171; font-weight: 600; padding: 1.5rem;'>⚠️ {resume_text}</div>",
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
        <div style='display: inline-block; width: 140px; height: 140px; border-radius: 50%; border: 10px solid rgba(255,255,255,0.05); border-top-color: {score_color}; position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.2);'>
            <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 2.2rem; font-weight: 800; font-family: "Outfit", sans-serif; color: #ffffff;'>{score}%</div>
        </div>
        <h3 style='margin: 10px 0 0 0; font-family: "Outfit", sans-serif; font-size: 1.2rem; color: #ffffff;'>ATS Compatibility Score</h3>
    </div>
    """
    
    # Matching skills tags HTML
    if matching:
        matching_html = "".join([f"<span style='display: inline-block; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; margin: 4px; font-weight: 500;'>{s}</span>" for s in matching])
    else:
        matching_html = "<span style='color: #94a3b8; font-size: 0.9rem;'>No matching skills identified.</span>"
        
    # Missing skills tags HTML
    if missing:
        missing_html = "".join([f"<span style='display: inline-block; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171; padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; margin: 4px; font-weight: 500;'>{s}</span>" for s in missing])
    else:
        missing_html = "<span style='color: #10b981; font-size: 0.9rem; font-weight: 600;'>🎉 No missing skills! Outstanding match.</span>"
        
    # Formatting bullet lists
    if formatting:
        formatting_md = "\n".join([f"- ⚠️ {item}" for item in formatting])
    else:
        formatting_md = "🟢 No significant formatting or structural issues found."
        
    # Recommendations bullet lists
    if recs:
        recs_md = "\n".join([f"- ✨ {item}" for item in recs])
    else:
        recs_md = "🟢 No immediate changes recommended. The resume is in excellent shape!"
        
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

# Custom styling with Dark Glassmorphism and modern colors
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Style root variables for Gradio elements */
:root, .gradio-container {
    --body-background-fill: radial-gradient(circle at top right, rgba(99, 102, 241, 0.15), transparent 45%),
                            radial-gradient(circle at bottom left, rgba(168, 85, 247, 0.15), transparent 45%),
                            #0b0f19 !important;
    --container-background-fill: rgba(17, 24, 39, 0.45) !important;
    --block-background-fill: rgba(17, 24, 39, 0.3) !important;
    --block-border-color: rgba(255, 255, 255, 0.08) !important;
    --block-border-width: 1px !important;
    --border-color-primary: rgba(255, 255, 255, 0.08) !important;
    --border-color-secondary: rgba(255, 255, 255, 0.05) !important;
    
    --input-background-fill: rgba(15, 23, 42, 0.6) !important;
    --input-border-color: rgba(255, 255, 255, 0.08) !important;
    --input-border-color-focus: #818cf8 !important;
    --input-placeholder-color: #64748b !important;
    
    --button-primary-background-fill: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%) !important;
    --button-primary-text-color: #ffffff !important;
    
    --button-secondary-background-fill: rgba(51, 65, 85, 0.5) !important;
    --button-secondary-background-fill-hover: rgba(71, 85, 105, 0.7) !important;
    --button-secondary-text-color: #f1f5f9 !important;
    
    /* Chatbot Bubble Colors */
    --chatbot-body-background-color: transparent !important;
    --message-user-background-color: #4f46e5 !important;
    --message-user-text-color: #ffffff !important;
    --message-user-border-color: rgba(99, 102, 241, 0.2) !important;
    --message-user-border-width: 1px !important;
    
    --message-bot-background-color: rgba(30, 41, 59, 0.7) !important;
    --message-bot-text-color: #f1f5f9 !important;
    --message-bot-border-color: rgba(255, 255, 255, 0.08) !important;
    --message-bot-border-width: 1px !important;
}

body, .gradio-container, input, button, textarea, select {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Header container styling */
#header-container {
    margin-bottom: 1.5rem !important;
    padding: 1.2rem 1.5rem !important;
    background: rgba(17, 24, 39, 0.45) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

/* Adjust primary buttons */
button.primary {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
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
            "<div style='display: flex; align-items: center; gap: 14px;'>"
            "<span style='font-size: 2.3rem;'>🔮</span>"
            "<div>"
            "<h1 style='margin: 0; font-size: 1.8rem; font-weight: 800; font-family: \"Outfit\", sans-serif; background: linear-gradient(135deg, #818cf8 10%, #a78bfa 50%, #f472b6 90%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>NaanChalant AI Resume ATS Analyzer</h1>"
            "<p style='margin: 2px 0 0 0; color: #94a3b8; font-size: 0.9rem; font-family: \"Plus Jakarta Sans\", sans-serif;'>Compare resumes against required skills with advanced metrics & get AI-powered rewrite recommendations.</p>"
            "</div>"
            "</div>"
        )

    with gr.Row():
        # Left Panel (Inputs)
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Resume & Requirements")
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
            gr.Markdown("### 📊 ATS Match Evaluation")
            score_widget = gr.HTML(
                value="<div style='text-align: center; color: #64748b; padding: 3rem;'><p style='font-size: 1.1rem;'>Upload a resume and paste job description to run ATS analysis.</p></div>"
            )
            
            with gr.Tabs():
                with gr.Tab("🔍 Skills & Keywords"):
                    gr.Markdown("#### Matching Skills (Found)")
                    matching_skills_widget = gr.HTML(value="<span style='color: #64748b; font-size: 0.95rem;'>No analysis performed yet.</span>")
                    gr.Markdown("#### Missing Skills (Gaps)")
                    missing_skills_widget = gr.HTML(value="<span style='color: #64748b; font-size: 0.95rem;'>No analysis performed yet.</span>")
                    gr.Markdown("#### Keyword Analysis")
                    keyword_analysis_widget = gr.Markdown(value="*No keyword analysis available.*")
                    
                with gr.Tab("📋 Recommendations"):
                    gr.Markdown("#### Actionable Edit Suggestions")
                    recommendations_widget = gr.Markdown(value="*Upload and analyze a resume to see actionable edit checklists.*")
                    
                with gr.Tab("🏗️ Formatting & Readability"):
                    gr.Markdown("#### Structural & Formatting Issues")
                    formatting_widget = gr.Markdown(value="*Upload and analyze a resume to detect structural formatting warnings.*")

    # Collapsible Corner Sidebar Chatbot
    with gr.Sidebar(label="💬 Resume Coach", open=False, position="right"):
        gr.Markdown("### 💬 Resume Coach")
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
            placeholder="<div style='text-align: center; padding-top: 2rem; color: #64748b;'><h4>💡 Ask optimization questions here!</h4></div>"
        )
        
        # Accordion for advanced model config inside sidebar
        with gr.Accordion("⚙️ Settings", open=False):
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