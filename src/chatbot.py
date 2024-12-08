import os
from dotenv import load_dotenv
from configobj import ConfigObj
import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from load_docs import load_docs_to_vectorStore as ldv
# import pprint
# from typing import Annotated
# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages


cfg = ConfigObj('./config.ini')
load_dotenv('env')
working_dir=os.path.dirname(os.path.abspath(__file__))

llm = ChatGroq(model=cfg['llm']['model'],
               temperature=cfg['llm']['temperature'],
               api_key=os.getenv('GROK_API_KEY'),
               )

def load_vectorStore():
    #define embedding
    embedding=HuggingFaceBgeEmbeddings(model_name=cfg["llm"]["embedding_model"])

    # vector store
    vector_db = Chroma(
        embedding_function=embedding,
        collection_name=cfg['vector_store']['collection_name'],
        persist_directory=cfg['vector_store']['persist_dir'],

    )
    return vector_db

def get_answer(query):
    retrived_result = vector_db.similarity_search(query=query)

    # format result
    retrived_result = [f"Result {i+1}: {item.page_content}" for i, item in enumerate(retrived_result)]

    prompt = PromptTemplate(
        template=cfg['template']['prompt'],
        input_variables=[cfg['template']['input_var1'], cfg['template']['input_var2']],
    )

    prompt = prompt.format(content=retrived_result, question=query)

    return llm.invoke(prompt)

st.set_page_config(
    page_title='PDF QnA',
    page_icon=':robot:',
    layout='centered',
)
st.title('QnA with documents')

#initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
uploaded_file=st.file_uploader(label="Upload text or PDF file", type=['pdf', 'txt'])

if uploaded_file:
    file_path=os.path.join(working_dir,'..',cfg['doc']['dir'],uploaded_file.name)
    with open(file_path, 'wb') as fp:
        fp.write(uploaded_file.getbuffer()) 
    
    if "vectorStore" not in st.session_state:
        st.session_state.vectorStore=ldv(file_path=file_path)
        
    if "load_vector_store" not in st.session_state:    
        vector_db = load_vectorStore()
        st.session_state.vector_db=vector_db
    
    print("Load vector store ended!")


for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_query = st.chat_input("Ask from pdf. ")

if user_query:
    st.session_state.chat_history.append({'role': 'Human', 'content': user_query})
    
    with st.chat_message('user'):
        st.markdown(user_query)

    with st.chat_message('AI'):
        ai_response = get_answer(user_query)
        # ai_response = response['answer']
        st.markdown(ai_response.content)
        st.session_state.chat_history.append({'role': 'AI', 'content': ai_response})

    
    
    
    