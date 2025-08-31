from App.constant import DATA_PATH,CHANK_SIZE,CHUNK_OVERLAP,EMEBEDDING_MODEL,CHROMA_DIR_PATH,LLM,PROMPT
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter 

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()

def get_embedding_model():
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    model_embedding = HuggingFaceEmbeddings(
    model_name=EMEBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )
    return model_embedding



def store_data():
    docs = TextLoader(DATA_PATH).load()
    docs = RecursiveCharacterTextSplitter(chunk_size=CHANK_SIZE,chunk_overlap=CHUNK_OVERLAP).split_documents(docs)
    chroma = Chroma(embedding_function=get_embedding_model(),persist_directory=CHROMA_DIR_PATH)
    chroma.add_documents(docs)
    

def get_llm_model():
    return ChatGroq(model=LLM,api_key=os.environ.get("GROK_API_KEY"))
    
def get_prompt():
    return PromptTemplate.from_template(PROMPT)

def get_retrever():
    chroma = Chroma(embedding_function=get_embedding_model(),persist_directory=CHROMA_DIR_PATH)
    return chroma.as_retriever(search_type='similarity',search_kwargs={'k':2})