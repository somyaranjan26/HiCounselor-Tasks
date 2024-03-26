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


# create embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    disallowed_special=()
)

vector_search = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

def query_data(query):
    docs = vector_search.similarity_search(query)
    as_output = docs[0].page_content
    
    llm = OpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )
    
    retriever = vector_search.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    
    retriever_output = qa.run(query)
    
    return retriever_output

with gr.Blocks(theme=Base(), title="RAG Logs QA App") as demo:
    gr.Markdown(
        """
        # RAG Logs QA App
        """
    )
    textbox = gr.Textbox(label="Enter a query")
    with gr.Row():
        submit = gr.Button("Submit", variant="primary", min_width="30px")
    with gr.Column():
        output = gr.Textbox(
            lines=1,
            max_lines=10,
            label="Output"
            )
    submit.click(query_data, inputs=textbox, outputs=output)

demo.launch()