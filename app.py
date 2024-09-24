import streamlit as st
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

import time
from dotenv import load_dotenv
#
# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
astra_db_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
api_key = os.getenv('NVIDIA_API_KEY')

# Initialize model and embeddings
# llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
# embeddings = NVIDIAEmbeddings()

llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

API_ENDPOINT = 'https://39690fd0-4733-43d1-853a-7dc382dc3d0a-us-east1.apps.astra.datastax.com'

# Vector store setup
vector_store = AstraDBVectorStore(
    collection_name="Analytics_Vidhya",
    embedding=embeddings,
    api_endpoint=API_ENDPOINT,
    token=astra_db_token,
)

# Vector embeddings function
def vector_embeddings():
    if 'vectors' not in st.session_state:
        # st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vector_store = AstraDBVectorStore(
            collection_name="Analytics_Vidhya",
            embedding=embeddings,
            api_endpoint=API_ENDPOINT,
            token=astra_db_token,
        )
        st.session_state.loader = WebBaseLoader(web_paths=('https://courses.analyticsvidhya.com/pages/all-free-courses',))
        st.session_state.docs =  st.session_state.loader.load()
        st.session_state.split = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        st.session_state.final_docs = st.session_state.split.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = st.session_state.vector_store.add_documents(st.session_state.final_docs)

# Page configuration
st.set_page_config(page_title='Smart Course Search - Analytics Vidhya', layout="centered")
st.title('Find the Perfect Free Course! 🎓')
st.caption('I am here to suggest you the best free courses on Analytics Vidhya!')

# Prompt template for LLM
prompt_template = '''
You are an intelligent assistant designed to help users find the most relevant free courses from Analytics Vidhya. Your task is to search through the provided course context and return concise, accurate, and relevant course or courses suggestions. 

Here are the instructions:
1. Use the context provided from the uploaded document to answer.
2. If you don’t know the answer or it is not in the context, clearly state "I don't know."
3. Keep your answers short and precise related to the question.

Context: {context}
User Question: {question}
Your Answer:
'''

prompt = ChatPromptTemplate.from_template(prompt_template)

if st.button('Start Session'):
    with st.spinner('Loading courses from Analytics Vidhya...'):
        vector_embeddings()
    st.success('Ask your query now!')

# Input and button for embedding
query = st.text_input('What do you want to learn today?', placeholder='Ask me about courses, e.g., "data science," "Python basics"')

# Running the query
if query:
    retriever = st.session_state.vector_store.as_retriever()
    chain = (
        {'context':retriever, 'question':RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    with st.spinner('Fetching the best courses for you...'):
        response = chain.invoke(query)
    st.write(response)

# End session and delete collection
if st.button('End Session'):
    vector_store.delete_collection()
    st.success('Session ended, and data cleaned up!')
