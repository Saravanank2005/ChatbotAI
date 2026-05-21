from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
import os
import json

# Load environment variables
load_dotenv()

# =========================
# LOAD DOCUMENTS (kept for reference, but not used in global chat)
# =========================

documents = []
folder_path = "notes"

if os.path.exists(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            loader = TextLoader(
                os.path.join(folder_path, file)
            )
            docs = loader.load()
            documents.extend(docs)
    print("Documents Loaded")
else:
    print(f"Warning: '{folder_path}' folder not found.")

# =========================
# SPLIT DOCUMENTS
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(
    documents
)

# =========================
# EMBEDDINGS
# =========================

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

texts = [
    doc.page_content
    for doc in split_docs
]

if texts:
    embeddings = embedding_model.encode(texts)
    print("Embeddings Created")
else:
    embeddings = []

# =========================
# CHROMADB
# =========================

client = chromadb.Client()

collection = client.create_collection(
    name="echomind"
)

if texts:
    for i, (text, embedding) in enumerate(
        zip(texts, embeddings)
    ):
        collection.add(
            documents=[text],
            embeddings=[embedding.tolist()],
            ids=[str(i)]
        )
    print("Stored in ChromaDB")

# =========================
# SEARCH FUNCTION (Local RAG)
# =========================

def search_notes(query, top_k=2):
    query_embedding = embedding_model.encode(
        [query]
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return results['documents'][0]

# =========================
# FILE PARSERS FOR RESUMES
# =========================

def extract_text_from_pdf(file_path):
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_path):
    try:
        import docx
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading Word document: {e}"

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        return f"Error reading text file: {e}"

def extract_resume_text(file_path):
    if not file_path:
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    else:
        return extract_text_from_txt(file_path)

# =========================
# ATS RESUME ANALYSIS
# =========================

def analyze_resume_ats(resume_text, required_skills, api_key=None):
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        return {
            "match_score": 0,
            "matching_skills": [],
            "missing_skills": [],
            "keyword_analysis": "Error: Gemini API key is missing.",
            "formatting_issues": [],
            "recommendations": ["Please configure GEMINI_API_KEY in your environment variables or space secrets."]
        }

    system_instruction = """You are an expert ATS (Applicant Tracking System) recruiter and resume optimization consultant.
Analyze the user's resume text against the provided required skills / job description.
Identify the match score, matching skills, missing skills, keyword gaps, and structural/formatting issues.
Provide highly actionable recommendations on how they can edit their resume to match the requirements better.

Return your response strictly as a JSON object with the following structure:
{
  "match_score": integer (0 to 100),
  "matching_skills": list of strings,
  "missing_skills": list of strings,
  "keyword_analysis": string (brief summary of keyword match quality),
  "formatting_issues": list of strings (structural flaws or formatting alerts),
  "recommendations": list of strings (actionable edit suggestions to improve the resume match)
}
Do not return any markdown code block wrap, HTML, or conversational filler outside the JSON."""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
            system_instruction=system_instruction
        )

        prompt = f"""### Required Skills / Job Description:
{required_skills}

### Resume Text:
{resume_text}
"""
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        analysis = json.loads(result_text)
        return analysis
    except Exception as e:
        return {
            "match_score": 0,
            "matching_skills": [],
            "missing_skills": [],
            "keyword_analysis": f"Error during analysis: {e}",
            "formatting_issues": [],
            "recommendations": [f"An error occurred while communicating with Gemini: {e}"]
        }

# =========================
# GLOBAL CHATBOT (No context limitation)
# =========================

def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    text_parts.append(part["text"])
                elif "content" in part:
                    text_parts.append(extract_text(part["content"]))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    elif isinstance(content, dict):
        if "text" in content:
            return extract_text(content["text"])
        elif "content" in content:
            return extract_text(content["content"])
    return str(content) if content is not None else ""

def chat_with_gemini(messages, api_key=None, model_name='gemini-2.5-flash', temperature=0.7, resume_text=None, required_skills=None):
    """
    Sends a conversation history list to Gemini and returns the response.
    messages format: [{"role": "user"|"assistant", "content": "string"}]
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        return "Error: Gemini API key is missing. Please configure GEMINI_API_KEY."

    try:
        # Search relevant notes if available (local templates / action verbs)
        raw_last_query = messages[-1]['content'] if messages else ""
        last_query = extract_text(raw_last_query)
        try:
            retrieved_docs = search_notes(last_query) if last_query else []
            context = "\n".join(retrieved_docs)
        except Exception as e:
            print(f"RAG search error (ignored): {e}")
            context = ""

        system_instruction = """You are NaanChalant AI, a premium conversational assistant.
You are helping the user optimize their resume for applicant tracking systems (ATS).
"""
        if resume_text and required_skills:
            system_instruction += f"""
Here is the user's resume text:
---
{resume_text}
---

And here are the required skills / job description they are targeting:
---
{required_skills}
---

Provide advice, rewrite bullet points using the STAR method, suggest bullet points for missing skills, and guide the user step-by-step on optimizing their resume for this role. Keep your answers formatting-friendly and easy to copy.
"""
        if context:
            system_instruction += f"\nAdditional retrieved general templates / guide notes:\n{context}\n"

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name,
            generation_config={"temperature": temperature},
            system_instruction=system_instruction
        )

        # Convert standard role list to Gemini structures
        contents = []
        for msg in messages:
            role = 'user' if msg['role'] == 'user' else 'model'
            msg_content = extract_text(msg['content'])
            contents.append({
                'role': role,
                'parts': [msg_content]
            })

        response = model.generate_content(contents)
        return response.text.strip()

    except Exception as e:
        return f"Error during Gemini generation: {e}"