import streamlit as st
from dotenv import load_dotenv

import os
import tempfile

from liteparse import LiteParse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# .env 파일 로드
load_dotenv()

# GROQ API KEY 가져오기
groq_api_key = os.getenv("GROQ_API_KEY")


# Streamlit 기본 설정
st.set_page_config(page_title="PDF RAG Chatbot")

# 웹 화면 제목
st.title("📄 PDF RAG Chatbot")


# PDF 업로드 기능
uploaded_file = st.file_uploader(
    "PDF 파일 업로드",
    type=["pdf"]
)


# 임베딩 모델 로드 함수
# cache_resource를 사용하면 한 번만 로드됨
@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"}
    )

    return embeddings


# PDF -> VectorStore 변환 함수
@st.cache_resource(show_spinner=False)
def create_vectorstore(pdf_path):

    # 임베딩 모델 불러오기
    embeddings = load_embeddings()

    # PDF 파서 생성
    parser = LiteParse()

    # PDF 파싱
    result = parser.parse(pdf_path)

    # PDF 전체 텍스트 추출
    pdf_text = result.text

    # LangChain Document 형식으로 변환
    docs = [
        Document(
            page_content=pdf_text,
            metadata={"source": pdf_path}
        )
    ]

    # 긴 문서를 chunk 단위로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    split_docs = text_splitter.split_documents(docs)

    # 벡터 DB 생성
    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )

    return vectorstore


# LLM 로드 함수
@st.cache_resource
def load_llm():

    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    return llm


# LLM 객체 생성
llm = load_llm()


# 프롬프트 템플릿
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


# 검색된 문서들을 문자열로 합치는 함수
def format_docs(docs):

    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


# 채팅 기록 저장용 session_state 생성
if "messages" not in st.session_state:

    st.session_state.messages = []


# PDF 업로드가 되었을 때만 실행
if uploaded_file is not None:

    # 업로드된 PDF를 임시 파일로 저장
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        tmp_pdf_path = tmp_file.name

    st.success("PDF 업로드 완료!")

    # VectorStore 생성
    with st.spinner("PDF 분석 중..."):

        vectorstore = create_vectorstore(tmp_pdf_path)

    # Retriever 생성
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # RAG Chain 생성
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 이전 채팅 기록 출력
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # 사용자 입력창
    if user_input := st.chat_input("질문을 입력하세요"):

        # 사용자 메시지 저장
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        # 사용자 메시지 화면 출력
        with st.chat_message("user"):

            st.markdown(user_input)

        # AI 응답 생성
        with st.chat_message("assistant"):

            with st.spinner("답변 생성 중..."):

                # RAG 실행
                response = rag_chain.invoke(user_input)

                # 응답 출력
                st.markdown(response)

        # AI 응답 저장
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

# PDF 업로드 전 안내 메시지
else:

    st.info("PDF 파일을 업로드해주세요.")