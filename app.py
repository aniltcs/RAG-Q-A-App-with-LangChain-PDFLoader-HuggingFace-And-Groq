import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="RAG Q&A App with directory having PDFs"

## load the GROQ API Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

## Huggingface embedding
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

prompt_template = """
You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{question}

Answer in clear and concise language:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## Data Ingestion step
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("RAG Document Q&A With Groq And Lama3")

user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

if user_prompt:
     retriever=st.session_state.vectors.as_retriever()

     # Retrieve relevant documents (chunks)
     relevant_docs = st.session_state.vectors.similarity_search(user_prompt)

    # Build chain
     retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
     start=time.process_time()
     response=retrieval_chain.invoke(user_prompt)
     
     print(f"Response time :{time.process_time()-start}")
     st.write(response)
     
     # Show the retrieved chunks
     with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(relevant_docs):
            st.write(doc.page_content)
            st.write("---")