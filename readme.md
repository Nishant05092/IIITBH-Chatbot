IIIT-BH Chatbot 🤖

A Retrieval-Augmented Generation (RAG) powered chatbot built using FastAPI, LangChain, Google Gemini, and ChromaDB, deployed on Render.

📌 Overview

This project is an intelligent chatbot designed for IIIT Bhagalpur information retrieval.
It leverages:

LLM (Google Gemini 1.5 Flash) for conversational capabilities

RAG (Retrieval-Augmented Generation) to provide accurate, document-aware answers

FastAPI backend with REST endpoints

ChromaDB for vector storage and retrieval

LangChain for building the retrieval + LLM pipeline

Custom frontend (HTML + CSS) for an interactive chat interface

Deployment on Render for public access

✨ Features

✅ RAG-powered responses → The chatbot answers queries based on an uploaded PDF (IIIT_Bhagalpur_Info.pdf).
✅ LLM-powered conversation → Uses Google Gemini 1.5 Flash for natural and contextual responses.
✅ Vector database → Stores and retrieves document embeddings via ChromaDB.
✅ FastAPI backend → Lightweight, fast, and scalable API service.
✅ Frontend (HTML/CSS) → Clean and responsive UI for users to interact with the bot.
✅ Deployed on Render → Accessible online with minimal setup.
✅ CORS-enabled API → Ready to integrate with external apps or services.

🛠️ Tech Stack

Backend: FastAPI, Python

LLM: Google Gemini 1.5 Flash (via langchain-google-genai)

Embeddings: Google Generative AI Embeddings (models/embedding-001)

Vector Database: ChromaDB

Document Loader: LangChain PyPDFLoader

Frontend: HTML, CSS

Deployment: Render

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/iiit-bh-chatbot.git
cd iiit-bh-chatbot

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set up Environment Variables

Create a .env file in the root directory and add:

GOOGLE_API_KEY=your_google_api_key_here

5️⃣ Run the Application Locally
uvicorn main:app --reload


The app will run at: http://127.0.0.1:8000/

📂 Project Structure
📦 iiit-bh-chatbot
├── main.py                # FastAPI backend + RAG pipeline
├── IIIT_Bhagalpur_Info.pdf # Knowledge base for chatbot
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
├── static/                 # Static assets (CSS, JS, Images)
├── templates/              # HTML templates (index.html)
├── vectorstore/            # Persistent ChromaDB embeddings
└── README.md               # Project documentation

🚀 Deployment on Render

Push the project to GitHub.

Create a new Render Web Service.

Connect your GitHub repo.

Add environment variable in Render dashboard:

GOOGLE_API_KEY=your_google_api_key_here

Use the following Start Command:

uvicorn main:app --host 0.0.0.0 --port 10000


Deploy → Render will assign you a live URL.

📡 API Endpoints
Root Endpoint
GET /


Returns the chatbot frontend (index.html).

Chat Endpoint
POST /chat


Request:

{
  "query": "Tell me about IIIT Bhagalpur departments"
}


Response:

{
  "response": "IIIT Bhagalpur offers B.Tech programs in CSE, ECE, and Mechatronics..."
}

🎨 Frontend Preview

Your frontend (index.html + static/styles.css) provides:

A chatbox UI with clean styling

User input field + submit button

Display of chatbot responses

(Insert screenshot here if available)

📖 Use Cases

📚 Student queries about IIIT Bhagalpur

🏫 Institute information bot for website

🤖 Demo for RAG-powered chatbots with LangChain

🚀 Template for building custom domain-specific bots

🧑‍💻 Contributing

Fork the repository

Create a feature branch (git checkout -b feature-name)

Commit your changes (git commit -m "Added new feature")

Push to the branch (git push origin feature-name)

Create a Pull Request
