from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
import os

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
# AI RAG AGENT (Local RAG)
# =========================

def ask_agent(query, api_key=None):
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        return "Error: Gemini API key is missing. Please set GEMINI_API_KEY in your .env file or enter it in the Streamlit app."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        retrieved_docs = search_notes(query)
        context = "\n".join(retrieved_docs)

        prompt = f"""You are NaanChalant AI Agent, a helpful assistant.
You have access to the user's personal notes below.

Retrieved Notes (use these if relevant to the question):
{context}

If the user is asking about their personal notes, answer based ONLY on the facts in the notes.
If the user's question is a general query (e.g. general programming, learning Java, facts, etc.) that has no relation to the personal notes, ignore the notes and answer the query fully and accurately using your general knowledge.

User Question:
{query}"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"Error during Gemini generation: {e}"

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

def chat_with_gemini(messages, api_key=None, model_name='gemini-2.5-flash', temperature=0.7):
    """
    Sends a conversation history list to Gemini and returns the response.
    messages format: [{"role": "user"|"assistant", "content": "string"}]
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        return "Error: Gemini API key is missing. Please configure GEMINI_API_KEY."

    try:
        # Search relevant notes if available
        raw_last_query = messages[-1]['content'] if messages else ""
        last_query = extract_text(raw_last_query)
        try:
            retrieved_docs = search_notes(last_query) if last_query else []
            context = "\n".join(retrieved_docs)
        except Exception as e:
            print(f"RAG search error (ignored): {e}")
            context = ""

        system_instruction = f"""You are NaanChalant AI, a premium conversational assistant.
You have semantic access to the user's local notes (simulating MCP file access).

Relevant retrieved snippets from local notes:
{context}

Guidelines:
- If the user is asking about their personal notes, activities, plans, or journal, answer based on the retrieved snippets.
- If the query is general (e.g. Java programming, coding, math, general knowledge), answer fully and accurately using your general intelligence. Ignore the retrieved notes if they are not relevant.
- Be conversational, helpful, and concise.
"""

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

# =========================
# CLI TEST
# =========================

if __name__ == "__main__":
    print("\n--- NaanChalant AI Global Agent CLI ---")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("\n[WARNING] GEMINI_API_KEY is not set.")
        user_key = input("Please enter your Gemini API Key: ").strip()
        if user_key:
            api_key = user_key

    chat_history = []
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        chat_history.append({"role": "user", "content": query})
        answer = chat_with_gemini(chat_history, api_key=api_key)
        chat_history.append({"role": "assistant", "content": answer})

        print("\nAI:", answer)