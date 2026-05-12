import gradio as gr

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -----------------------------------
# 환경 변수 로드
# -----------------------------------
load_dotenv()

# -----------------------------------
# PDF 로드
# -----------------------------------
loader = PyPDFLoader("./AI브리프_3월_260303.pdf")
docs = loader.load()

# -----------------------------------
# 문서 분할
# -----------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(docs)

# -----------------------------------
# 임베딩 모델
# Ollama 실행 필요:
# ollama serve
# ollama pull bge-m3
# -----------------------------------
embeddings = OllamaEmbeddings(
    model="bge-m3"
)

# -----------------------------------
# 벡터 DB
# persist_directory 추가 추천
# -----------------------------------
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# -----------------------------------
# LLM
# -----------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_retries=2,
)

# -----------------------------------
# Prompt
# -----------------------------------


template = """
당신은 문서 검색 기반 AI 어시스턴트입니다.

아래 제공된 Context만 이용하여 답변하세요.

절대 지켜야 할 규칙:
- Context에 없는 내용은 추측하지 마세요.
- 불확실하면 답변하지 마세요.
- 일반 지식을 사용하지 마세요.
- 답을 찾을 수 없으면 아래 문장을 그대로 출력하세요.

"제공된 문서에서 해당 내용을 찾을 수 없습니다."

답변은 한국어로 간결하게 작성하세요.

[Context]
{context}

[Question]
{question}

[Answer]
"""



prompt = ChatPromptTemplate.from_template(template)

# -----------------------------------
# 문서 formatting
# -----------------------------------
def format_docs(docs):
    return "\n\n".join(
        [doc.page_content for doc in docs]
    )

# -----------------------------------
# RAG Chain
# -----------------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------------
# 채팅 함수
# -----------------------------------
def chat(message, history):

    if not message.strip():
        return "질문을 입력해주세요."

    try:
        response = rag_chain.invoke(message)
        return response

    except Exception as e:
        return f"에러 발생:\n{str(e)}"

# -----------------------------------
# Gradio UI
# -----------------------------------
demo = gr.ChatInterface(
    fn=chat,
    title="📚 PDF 기반 RAG 챗봇",
    description="PDF 문서를 기반으로 답변합니다.",
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(
        placeholder="질문을 입력하세요...",
        container=True,
        scale=7,
    ),
)

# -----------------------------------
# 실행
# -----------------------------------
demo.launch()