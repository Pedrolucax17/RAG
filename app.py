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


