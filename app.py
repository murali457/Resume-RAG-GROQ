import os
from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.rate_limiters import InMemoryRateLimiter # NEW: Rate Limiter import
import google.generativeai as genai


from dotenv import load_dotenv
load_dotenv()

# Setup caching to save your API quota for repeated questions
set_llm_cache(InMemoryCache())

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- CONFIGURATION ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# REMINDER: Secure this key in an environment variable later!



os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 1. NEW: Initialize Rate Limiter
# 0.25 requests per second = 15 requests per minute (Standard Gemini Free Tier)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.25, 
    check_every_n_seconds=0.1, 
    max_bucket_size=1
)

# 2. Initialize Components
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0, 
    max_retries=6,
    rate_limiter=rate_limiter # NEW: Integrated Rate Limiter
)

vector_db = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ingest', methods=['POST'])
def ingest():
    global vector_db
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    try:
        # Step 1: Load & Split
        loader = PyPDFLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # Step 2: Create local Vector Store
        vector_db = FAISS.from_documents(splits, embeddings)
        vector_db.save_local("vector_store")
        
        return jsonify({"message": f"Successfully ingested {len(splits)} chunks."})
    except Exception as e:
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500
  
@app.route('/suggest', methods=['POST'])
def suggest():
    global vector_db
    if not vector_db:
        return jsonify({"questions": []})

    # 1. Grab specifically the top summary/experience chunks
    docs = vector_db.similarity_search("Summary of professional experience and key skills", k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Strategic Prompting for Resumes/Professional Docs
    suggestion_template = """
    You are a Senior Recruiter analyzing a candidate's resume. 
    Based on the context below, generate 10 professional, specific questions that 
    a hiring manager would actually ask this person. 
    
    Avoid general theories. Focus on:
    - Specific projects mentioned in the text.
    - Years of experience with specific tools (e.g., Python, AWS).
    - Achievements or certifications found in the document.
    
    Format: Just a list of 10 questions, one per line. No headers.
    
    Context: {context}
    """
    
    try:
        response = llm.invoke(suggestion_template.format(context=context))
        # Split by newline and remove numbering if the LLM added it
        questions = [q.strip().lstrip('1234567890. -') for q in response.content.split('\n') if len(q.strip()) > 10]
        return jsonify({"questions": questions[:10]})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/ask', methods=['POST'])
def ask():
    global vector_db
    query = request.json.get('question')
    
    # Load from disk if global variable is lost
    if not vector_db and os.path.exists("vector_store"):
        vector_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

    if not vector_db:
        return jsonify({"answer": "Please ingest a PDF first."})

    # Step 3: Modern RAG Chain (LCEL)
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        # Rate Limiter automatically handles the timing here
        response = rag_chain.invoke(query)
        return jsonify({"answer": response})
    
    except Exception as e:
        if "429" in str(e):
            return jsonify({
                "answer": "Daily/Minute API limit reached. The system is waiting to stay within free tier limits. Please wait a few seconds and try again."
            })
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)