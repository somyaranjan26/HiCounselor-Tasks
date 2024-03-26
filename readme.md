## RagLogs QA App
App allow to question your logs files for better understanding of production app behaviour.

___
### load_data.py
This code snippet connects to a MongoDB Atlas cluster, loads text data from a directory, splits the text into smaller chunks, creates embeddings using OpenAI, and performs vector search using MongoDB Atlas. 

The code snippet performs the following steps:
1. Imports necessary libraries and modules.
2. Loads environment variables from a .env file.
3. Connects to a MongoDB Atlas cluster using the provided URI.
4. Defines the names of the MongoDB database, collection, and vector search index.
5. Initializes a MongoDB collection object.
6. Loads text data from a directory using a DirectoryLoader.
7. Splits the text into smaller chunks using a RecursiveCharacterTextSplitter.
8. Creates embeddings using OpenAIEmbeddings.
9. Performs vector search using MongoDBAtlasVectorSearch.

### app.py
This code snippet is a Python script that creates a Gradio interface for a question-answering application using the RAG (Retrieval-Augmented Generation) model. 

The script imports necessary libraries and modules, including vector stores, embeddings, document loaders, and the RAG model implementation. It also connects to a MongoDB Atlas cluster to retrieve data for the question-answering process.

The `query_data` function takes a query as input and performs a similarity search using the MongoDB Atlas vector search. It then uses the RAG model to generate an answer based on the retrieved documents. The function returns the generated answer.

The script also creates a Gradio interface with a textbox for entering queries and a submit button. When the submit button is clicked, the `query_data` function is called with the input query, and the generated answer is displayed in the output textbox.

To run the script, the necessary environment variables (MongoDB URI and OpenAI API key) should be set in a .env file.
