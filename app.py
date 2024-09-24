import streamlit as st
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')


llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def vector_embeddings():
    if 'vector_store' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        loader = WebBaseLoader(web_paths=('https://courses.analyticsvidhya.com/pages/all-free-courses',))
        docs = loader.load()
        split = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        final_docs = split.split_documents(docs[:30])

        persist_directory = 'chroma_db'
        st.session_state.vector_store = Chroma.from_documents(
            documents=final_docs,
            embedding=st.session_state.embeddings,
            persist_directory=persist_directory
        )
        st.session_state.vector_store.persist()


st.set_page_config(page_title='Smart Course Search - Analytics Vidhya', layout="centered")
st.title('Find the Perfect Free Course! 🎓')
st.caption('I am here to suggest you the best free courses on Analytics Vidhya!')


prompt_template = '''
You are an intelligent assistant designed to help users find the most relevant free courses from Analytics Vidhya. Your task is to search through the provided course context and return concise, accurate, and relevant course or courses suggestions. 
Here are the instructions:
1. Use the context provided from the uploaded document to answer.
2. If you don't know the answer or it is not in the context, clearly state "I don't know."
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


query = st.text_input('What do you want to learn today?', placeholder='Ask me about courses, e.g., "data science," "Python basics"')


if query and 'vector_store' in st.session_state:
    retriever = st.session_state.vector_store.as_retriever()
    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    with st.spinner('Fetching the best courses for you...'):
        response = chain.invoke(query)
    st.write(response)


if st.button('End Session'):
    if 'vector_store' in st.session_state:
        st.session_state.vector_store.delete_collection()
        del st.session_state.vector_store
    st.success('Session ended, and data cleaned up!')
