from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from configobj import ConfigObj
import os 
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

load_dotenv('env')
cfg = ConfigObj("./config.ini")

# Load docs

def load_docs_to_vectorStore(file_path):
    # f_name = cfg["doc"]["f_name"]
    # f_dir = cfg["doc"]["dir"]
    # file_path = os.path.join( os.getcwd(),f_dir,f_name)
    if file_path.endswith('.pdf'):
        loader = UnstructuredPDFLoader(file_path=file_path)
    else:
        loader = TextLoader(file_path=file_path)
    loader = TextLoader(file_path)
    doc = loader.load()
    # print(doc)

    # Chunking
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(cfg['vector_store']['chunk_size']), 
        chunk_overlap=int(cfg['vector_store']['chunk_overlap']))
    chunked_docs = txt_splitter.split_documents(doc)
    # pprint.pp(chunked_docs)

    # Embedding
    embedding = HuggingFaceBgeEmbeddings(model_name=cfg['llm']['embedding_model'])

    # vector store initialize    
    vector_db = Chroma.from_documents(
        documents=chunked_docs,
        collection_name=cfg['vector_store']['collection_name'],
        embedding=embedding,
        persist_directory=cfg['vector_store']['persist_dir'])

    print("Docs loaded into vector store successfully!")
    # query='disappeared into fog'
    # retrived_result = vector_db.similarity_search(query=query)
    # print(retrived_result)
    # pprint.pp(retrived_result)



