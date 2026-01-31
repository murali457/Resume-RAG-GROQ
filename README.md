# 🤖 Resume RAG: Intelligent PDF Intelligence Assistant

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![AI-Model](https://img.shields.io/badge/LLM-Llama--3.3--70B-orange.svg)](https://groq.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-FAISS-green.svg)](https://github.com/facebookresearch/faiss)

A high-performance **Retrieval-Augmented Generation (RAG)** application designed to transform static resumes into interactive, searchable intelligence. This project leverages **Groq's Llama 3.3** for near-instant inference and **HuggingFace BGE Embeddings** for precise semantic retrieval.

---

## 📺 Preview

### **The Intelligent Dashboard**
![Ready State Interface](screenshots/home_page.png)
*Professional, dark-mode UI with sidebar navigation and contextual ready-states.*

### **Real-time Knowledge Extraction**
![Active Chat Interface](screenshots/results_page.png)
*AI Assistant generating "Smart Queries" and providing deep "AI Insights" based on resume content.*

---

## ✨ Key Features

- **🚀 Lightning-Fast Inference:** Utilizes Groq Cloud LPUs to achieve high-token-per-second responses.
- **🧠 Semantic Memory:** Implements conversation history to handle complex, multi-turn follow-up questions.
- **📂 Local Vector Indexing:** Uses FAISS and BGE-Small-En for lightweight, local vectorization—no external database costs.
- **🪄 Smart Query Generation:** Automatically analyzes the uploaded PDF to suggest the top 10 most relevant interview questions.
- **⚡ Async Processing:** Real-time loaders and feedback ensure a smooth, enterprise-grade user experience.

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Orchestration:** LangChain (LCEL)
- **LLM:** Llama 3.3 70B (via Groq)
- **Embeddings:** HuggingFace `BAAI/bge-small-en-v1.5`
- **Vector Store:** FAISS
- **Frontend:** Tailwind CSS, JavaScript (ES6+)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or higher
- A Groq API Key ([Get one here](https://console.groq.com/))

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/murali457/Resume-RAG-GROQ.git](https://github.com/murali457/Resume-RAG-GROQ.git)
cd Resume-RAG-GROQ

# Install dependencies
pip install -r requirements.txt

------------------------------------------------------------------

## 👨‍💻 Author
**Murali Koppula** *Full Stack Python Developer | AI Enthusiast*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/murali-koppula06a1ba295)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/murali457)