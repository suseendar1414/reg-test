from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.llms import OpenAI

client = MongoClient(key_param.MONGO_URI)
dbName = "langchain"
collectionName = "collection of text blobs"
collection = client[dbName][collectionName]

loader = DirectoryLoader('./sample_files',glob="./*.txt",show_progress = True)
data = loader.load()


embeddings = OpenAIEmbeddings(openai_api_key=key_param.open_api_key)

vectorStore = MongoDBAtlasVectorSearch.from_documents( data, embeddings, collection = collection )