import gradio as gr
import os
from dotenv import load_dotenv
from rag_pipeline import chat_with_gemini, extract_text

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

load_dotenv()

def chat_response(message, history, model_dropdown, temperature_slider):
    """
    Callback function for Gradio ChatInterface.
    history is a list of ChatMessage objects.
    """
    # Create the conversation list to send to Gemini
    formatted_messages = []
    
    # Map previous conversation history
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
        
        # Extract clean text to avoid Gradio 6 nested lists/dicts crashing the pipeline or search
        content_str = extract_text(content)
        
        # In Gradio, role can be 'user' or 'assistant'.
        # We pass this role to chat_with_gemini which maps it to 'user' or 'model'.
        formatted_messages.append({"role": role, "content": content_str})
        
    # Append the new message
    message_str = extract_text(message)
    formatted_messages.append({"role": "user", "content": message_str})
    
    # Fetch the API key from environment / .env within project itself
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key.strip() in ("", "YOUR_GEMINI_API_KEY"):
        return "Error: Gemini API key is missing. Please configure GEMINI_API_KEY in your .env file or Hugging Face Space secrets."
        
    try:
        response = chat_with_gemini(
            formatted_messages, 
            api_key=api_key, 
            model_name=model_dropdown, 
            temperature=temperature_slider
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

/* API status box styling */
.api-status-box {
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 15px;
}
.api-status-box.active {
    background-color: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
    color: #34d399 !important;
}
.api-status-box.warning {
    background-color: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.25) !important;
    color: #f87171 !important;
}

/* Chatbot container styles */
#chatbot-view {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Adjust primary and secondary buttons */
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

with gr.Blocks(title="NaanChalant AI Hub") as demo:
    # Sleek low-profile header row
    with gr.Row(elem_id="header-container"):
        gr.HTML(
            "<div style='display: flex; align-items: center; gap: 14px;'>"
            "<span style='font-size: 2.3rem;'>🔮</span>"
            "<div>"
            "<h1 style='margin: 0; font-size: 1.8rem; font-weight: 800; font-family: \"Outfit\", sans-serif; background: linear-gradient(135deg, #818cf8 10%, #a78bfa 50%, #f472b6 90%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>NaanChalant AI Hub</h1>"
            "<p style='margin: 2px 0 0 0; color: #94a3b8; font-size: 0.9rem; font-family: \"Plus Jakarta Sans\", sans-serif;'>Your premium assistant powered by Google Gemini API & local semantic RAG.</p>"
            "</div>"
            "</div>"
        )

    # Use native Sidebar layout which collapses automatically
    with gr.Sidebar(label="Chat Settings", open=True):
        gr.Markdown("### ⚙️ Chat Settings")
        
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
        
        gr.Markdown("---")
        gr.Markdown(
            "💬 **Global Capabilities**\n"
            "- Full multi-turn dialog memory\n"
            "- Freeform querying (math, coding, design, reasoning)\n"
            "- Local RAG pipeline loaded in backend context\n\n"
            "🔑 *API key is loaded directly from project configuration (.env or Space Secrets).*"
        )
        
    # Chat interface panel (taking the remaining workspace width)
    chatbot = gr.Chatbot(
        height=520,
        show_label=False,
        elem_id="chatbot-view",
        layout="bubble",
        placeholder="<div style='text-align: center; padding-top: 4rem; color: #64748b;'><h3>✨ Welcome to NaanChalant AI</h3><p>Type your query below to begin!</p></div>"
    )
    
    gr.ChatInterface(
        fn=chat_response,
        chatbot=chatbot,
        additional_inputs=[model_dropdown, temperature_slider],
        examples=[
            ["Help me learn Java from scratch.", "gemini-2.5-flash", 0.7],
            ["Explain what is a Retrieval-Augmented Generation (RAG) system.", "gemini-2.5-flash", 0.7],
            ["Write a python function to find the shortest path in a graph.", "gemini-2.5-flash", 0.7]
        ]
    )

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Soft(),
        css=custom_css
    )