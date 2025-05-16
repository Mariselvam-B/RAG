# 🤖 AI Chatbot with PostgreSQL + Ollama (RAG Implementation)

A cost-free, open-source AI chatbot that uses your **own business data** to provide smart, context-aware responses — built with:

- 🗂️ **PostgreSQL** for storing and retrieving documents  
- 🧠 **Ollama** to run local LLMs (like Mistral) privately and efficiently  
- 🔁 **Retrieval-Augmented Generation (RAG)** to combine data retrieval and AI generation

No cloud APIs. No recurring fees. 100% local.

---

## 🚀 Features

- Full RAG pipeline using open-source tools  
- Private and secure — all data stays on your machine  
- Flexible: customize for internal support, customer service, or knowledge-based assistants  
- Simple to set up and extend

---

## 🛠️ Installation & Setup

### 1. Clone the repository

Install Ollama - https://ollama.com/download

```bash
git clone https://github.com/Mariselvam-B/RAG.git
cd RAG
pip install -r requirements.txt
ollama pull mistral
python app.py
```
