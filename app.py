import os
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

caminho_pdf = "Perceptron.pdf"
loader = PyPDFLoader(caminho_pdf) # Instancia a classe para preparar para o carregamento.
documentos = loader.load() # Carrega o documento


def train():
  splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  chunks = splitter.split_documents(documentos)
  
  embeddings = OpenAIEmbeddings()
  db_path = "banco-faiss"
  
  if os.path.exists(db_path):
    vectordb = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    vectordb.add_documents(chunks)
  else:
    vectordb = FAISS.from_documents(chunks, embeddings)
  
  vectordb.save_local(db_path)


train()