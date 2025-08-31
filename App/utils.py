from App.constant import EMEBEDDING_MODEL
from App.constant import DATA_PATH,CHANK_SIZE,CHUNK_OVERLAP
from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter 


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
    chroma = Chroma(embedding_function=get_embedding_model(),persist_directory='./chroma')
    chroma.add_documents(docs)
    chroma.persist()


