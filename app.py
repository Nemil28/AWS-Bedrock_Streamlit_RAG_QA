import json
import os
import sys
import boto3
import streamlit as st

## Use Amazon Titan Embedding model to generate Embeddings

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Vector Embedding and Vector Store

from langchain.vectorstores import FAISS 

## LLM models Prompt Template

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



bedrock = boto3.client(service_name = 'bedrock-runtime')

bedrock_embedding = BedrockEmbeddings(model_id = 'amazon.titan-embed-text-v1', client = bedrock)

## Data Ingestion

def data_ingestion():
    loader = PyPDFDirectoryLoader('data')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000,
        chunk_overlap = 1000,
    )

    docs = text_splitter.split_documents(documents)
    
    return docs

## Vector Embeddings and Vector Store

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embedding
    )

    vectorstore_faiss.save_local('faiss_index')

## Create AI21 Jurassic Model
    
def get_jurassic_llm():
    llm = Bedrock(
        model_id = 'ai21.j2-mid-v1',
        client = bedrock,
        model_kwargs = {'maxTokens': 512}
    )

    return llm

## Create Meta Llama2 Model
    
def get_llama2_llm():
    llm = Bedrock(
        model_id = 'meta.llama2-70b-chat-v1',
        client = bedrock,
        model_kwargs = {'max_gen_len': 512}
    )

    return llm

## Prompt Template

prompt_template = """

Human: Use the following pieces of context to provide a concise answer
to the question at the end but use at least 250 words to summarize with
detailed explanations. If you don't know the answer, just say that you don't
know, don't try to make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant: 

"""

PROMPT = PromptTemplate(
    template = prompt_template,
    input_variables = ['context', 'question']
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = vectorstore_faiss.as_retriever(
            search_type = 'similarity',
            search_kwargs = {'k': 3}
        ),
        return_source_documents = True,
        chain_type_kwargs = {'prompt': PROMPT}
    )

    answer = qa({'query': query})

    return answer['result']

def main():
    st.set_page_config('Chat PDF for CMPE 258')
    st.header('Chat with the study material for CMPE 258 using AWS Bedrock🔎')

    user_question = st.text_input("Ask anything you'd like to know from the study material PDFs")

    with st.sidebar:
        st.title('Update or create Vector Store')

        if st.button('Vector Update'):
            with st.spinner('Processing...'):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success('Done.')

    if st.button('AI21 Jungle Output'):
        with st.spinner('Processing...'):
            faiss_index = FAISS.load_local('faiss_index', bedrock_embedding)
            llm = get_jurassic_llm()

            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success('Done')

    if st.button('Meta Llama2 Output'):
        with st.spinner('Processing...'):
            faiss_index = FAISS.load_local('faiss_index', bedrock_embedding)
            llm = get_llama2_llm()

            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success('Done')

if __name__ == '__main__':
    main()