import json
import os
import sys

import streamlit as st
import shutil
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

# import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


## Gemini Clients

load_dotenv()
api_key = os.environ.get('api_key')
## We will be suing Titan Embeddings Model To generate Embedding

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA




## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        embeddings
    )

    vectorstore_faiss.save_local("faiss_index")

def get_gemini_llm():
  
    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=api_key,
                             temperature=0.1,convert_system_message_to_human=True)
    
    return model



prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(model,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']




import os
import streamlit as st
import shutil
import time

class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def main():
    st.set_page_config("Ask your PDF")
    
    st.header("Doc-Query - Interact with Your Documents")
    
    st.image("assets/bot_read.jpg", use_column_width=True)

    session_state = SessionState (upload_button_clicked=False,file_path=None)
    user_question = ''
    

    # Move the content to the sidebar after processing
    with st.sidebar:
        st.title("Upload the PDF")
        file = st.file_uploader("", type=["pdf"])
    
        # Save file to folder
        if file is not None:
            # Create a folder named "data" if it doesn't exist
            if not os.path.exists("data"):
                os.makedirs("data")
            else:
                shutil.rmtree("data")
                os.makedirs("data")
            
            # Save the uploaded file to the "data" folder
            session_state.file_path = os.path.join("data", file.name)
            with open(session_state.file_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"File saved to: {session_state.file_path}")


                
                

        if not session_state.upload_button_clicked:
            if st.button("Upload"):    
                with st.spinner("Processing..."):
                    docs = data_ingestion()
                    get_vector_store(docs)
                    session_state.upload_button_clicked=True
                    st.success("Done")
            
            if st.button("Delete and Upload new file"):
                    if session_state.file_path:
                        os.remove(session_state.file_path)
                        st.success("File deleted successfully!")
                    else:
                        st.warning("No file uploaded to delete.")
                    session_state.upload_button_clicked = False

    

    st.write("""
1. **Upload Your Documents:** Select one or multiple PDF files and upload them to the platform.
2. **Click on Upload:** After selecting your documents from your device, click the upload button to proceed.
3. **Interact with Your Documents:**  Ask questions from your document.
""")
    user_question = st.text_input("Ask a Question")

    if st.button("Send"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            llm = get_gemini_llm()
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")
        
                        



if __name__ == "__main__":
    main()















