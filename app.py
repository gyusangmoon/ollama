import streamlit as st
from dotenv import load_dotenv

from liteparse import LiteParse

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_experimental.text_splitter import SemanticChunker
#from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

import os
import re


# env 불러오기
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


st.title("📄 PDF RAG Chatbot")


@st.cache_resource
def load_pdf():

    parser = LiteParse()

    result = parser.parse("./AI브리프_3월_260303.pdf")

    pdf_text = result.text

    # 줄바꿈 정리
    #pdf_text = re.sub(r"\n+", " ", pdf_text)

    return pdf_text


# EMBEDDING + CHUNKING

@st.cache_resource
def create_vectorstore():

    pdf_text = load_pdf()

   #embeddings = HuggingFaceInferenceAPIEmbeddings(
   #     api_key=hf_token,
   #     model_name="BAAI/bge-m3"
   # )
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=50,
    )

    docs = text_splitter.create_documents([pdf_text])

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore

vectorstore = create_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# LLM

@st.cache_resource
def load_llm():

    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0,
        max_retries=2,
    )

    return llm

llm = load_llm()

# PROMPT

template = """
Answer the question based only on the following context.

[Context]
{context}

[Question]
{question}

[Answer (in Korean)]
"""

prompt = ChatPromptTemplate.from_template(template)

# FORMAT DOCS

def format_docs(docs):
    return "\n\n".join(
        [doc.page_content for doc in docs]
    )

# RAG CHAIN

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# CHAT HISTORY

if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 출력
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# USER INPUT

if user_input := st.chat_input("질문을 입력하세요"):

    # 사용자 메시지 저장
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # 사용자 메시지 출력
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 응답 생성
    with st.chat_message("assistant"):

        with st.spinner("답변 생성 중..."):

            response = rag_chain.invoke(user_input)

            st.markdown(response)

    # 응답 저장
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )