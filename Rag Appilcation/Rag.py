import streamlit as st

# Import Libraries
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Import Libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env file into environment

groq_api_key = os.getenv("GROQ_API_KEY")


# Setup ui 
st.title("Rag Chatbot")
# setup the session state messages to hold the old messages 
if 'messages' not in st.session_state:
    st.session_state.messages=[]

# Display all the historical messages 
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])  

@st.cache_resource
def get_vectorstore():
    pdf_name = "Student-Hand-Book-updated-Final.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    # Create chunks aka vectors (chromadb)
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name = "all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 100)
    ).from_loaders(loaders)
    return index.vectorstore

prompt = st.chat_input("Pass your prompt here")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
        the most accurate and most precise answers. Answer the following Question: {user_prompt}.
        Start the answer directly. No small talk please

        """)
    model = "llama3-8b-8192"
    groqchat = ChatGroq(
        groq_api_key="YOUR API KEY",
        #groq_api_key = os.environ.get("GROQ_API_KEY"),
        model_name=model
    ) 
    try:
        vector_store = get_vectorstore()
        if vector_store is None:
            st.error("Failed to load the document")

        chain = RetrievalQA.from_chain_type(
            llm = groqchat,
            chain_type = 'stuff',
            retriever = vector_store.as_retriever(search_kwargs =({"k":3})),
            return_source_documents=True
        )
        result = chain({"query":prompt})
        response = result["result"]
         
        st.chat_message("user").markdown(response)
        st.session_state.messages.append({'role':'user','content':response})
    except Exception as e:
        st.error("Error: " + str(e))
