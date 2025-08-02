# Python Backend (FastAPI) - main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not loaded. Check your .env file and key name.")

# Document loading and preprocessing
loader = PyPDFLoader("IIIT_Bhagalpur_College_Info.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Embedding and vector store
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./vectorstore"
)

# Chat model and retrieval chain
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=api_key
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False
)

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Chat endpoint
class Query(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: Query):
    try:
        result = qa_chain.invoke({"query": query.query})
        return {"response": result["result"]}
    except Exception as e:
        return {"error": str(e)}