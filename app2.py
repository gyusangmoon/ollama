import streamlit as st
from dotenv import load_dotenv

import os

from liteparse import LiteParse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# Streamlit UI

st.set_page_config(page_title="PDF RAG Chatbot")

st.title("📄 PDF RAG Chatbot")


# Embedding 모델

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"}
    )

    return embeddings


# VectorStore 로드 또는 생성

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    db_path = "./db/faiss"

    # 기존 DB 존재하면 로드


    if os.path.exists(db_path):

        st.info("기존 FAISS DB 로드 중...")

        vectorstore = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    # 없으면 새로 생성

    st.info("FAISS DB 생성 중...")

    parser = LiteParse()

    result = parser.parse("AI브리프_3월_260303.pdf")

    pdf_text = result.text

    docs = [
        Document(
            page_content=pdf_text,
            metadata={"source": "AI브리프_3월_260303.pdf"}
        )
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    split_docs = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )

    # 저장
    vectorstore.save_local(db_path)

    return vectorstore

# Retriever

vectorstore = load_vectorstore()

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
    )

    return llm


llm = load_llm()

# Prompt

template = """
다음 Context만 기반으로 질문에 답변하세요.
모르는 답변은 pdf에 없는 정보라 모른다고 답해주세요.

[Context]
{context}

[Question]
{question}

[Answer]
"""

prompt = ChatPromptTemplate.from_template(template)


# 문서 포맷 함수

def format_docs(docs):

    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


# RAG Chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# 채팅 기록

if "messages" not in st.session_state:

    st.session_state.messages = []


# 기존 메시지 출력
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# 사용자 입력

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