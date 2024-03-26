from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from gradio.themes.base import Base
from pymongo import MongoClient
from dotenv import load_dotenv
import gradio as gr
import os


# Load environment variables from .env file
load_dotenv()


# Connect to MongoDB Atlas cluster
client = MongoClient(os.getenv("MONGO_URI"))

DB_NAME = "RagLogsDatabase"
COLLECTION_NAME = "RagLogsCollection"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]


# load the directory
loader = DirectoryLoader('./logs', glob="./*.txt", show_progress=True)
data = loader.load()


# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
texts = text_splitter.split_documents(data)

# create embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    disallowed_special=()
)


vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=texts,
    embedding=embeddings,
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)
