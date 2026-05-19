'''
pip install gradio python-dotenv langchain langchain-core
 langchain-community langchain-text-splitters 
 langchain-chroma langchain-groq langchain-ollama pypdf
'''

# python에서 pdf 파일을 다룰 때 가장 기본적인 순수 파이썬 라이브러리라 사용
from langchain_community.document_loaders import PyPDFLoader
# Recursive Splitter가 긴 텍스트, 문맥을 유지하고 싶을 때 더 효율적
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ollama가 로컬환경에서 llm을 쉽게 실행할 수 있게 해주어서 사용
from langchain_ollama import OllamaEmbeddings
# faiss db는 속도가 빠르지만 chromadb는 영속적으로 데이터를 저장할 수 있어서 사용
from langchain_chroma import Chroma
# Groq이 무료 api 키를 제공해주고 있기 때문에 사용
from langchain_groq import ChatGroq
# ChatPromptTemplate이 메시지 기반의 대화 흐름을 구성하는데 유용해서 사용
from langchain_core.prompts import ChatPromptTemplate
# StrOutputParser는 LLM에서 나오는 언어모델의 출력을 문자열 형식으로 변환시키기 위해 사용
from langchain_core.output_parsers import StrOutputParser
# invoke 메소드를 통해 입력된 데이터를 그대로 반환, 전달하기 위해 사용
from langchain_core.runnables import RunnablePassthrough

import gradio as gr

# env 환경설정을 불러오기 위해 사용
from dotenv import load_dotenv 

load_dotenv()

# pypdfLoader 를 활용하여 pdf를 불러옴.
loader = PyPDFLoader("./AI브리프_3월_260303.pdf")
# load 함수를 이용하여 불러온 pdf 파일을 document 객체로 변환후 docs 에 저장
docs = loader.load()

# chunkviz라는 사이트에서 설정값 별 나뉘는 단위 확인 가능
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# split_documents 메소드를 활용하여 분할 후 split_docs에 저장
split_docs = text_splitter.split_documents(docs)

# bge-m3 모델이 한국어를 포함한 다국어 기능을 잘 제공해준다하여 사용
embeddings = OllamaEmbeddings(
    model="bge-m3"
)

# chroma db의 from_documents 클래스 메소드를 통해 vector store를 생성
vectorstore = Chroma.from_documents(
    documents=split_docs,  # 백터 저장소에 추가할 문서 리스트를 split_docs로 지정
    embedding=embeddings,  # 임베딩할 함수를 아까 지정해주었던 embeddings로 설정
    persist_directory="./chroma_db" # 매 번 임베딩하면 속도가 느려지므로 영구적으로 컬렉션을 저장할 디렉토리를 ./chroma_db로 설정
)

# retriever는 RAG 시스템에서 정보 검색의 질을 결정하는 핵심적인 역할을 함.
# retriever과 벡터 스토어의 다른점은 벡터스토어는 데이터를 저장하는 것이고, retriever는 데이터를 검색하는 것이다.

# as_retriever 메소드를 통해 검색 옵션을 설정하여 요구에 맞는 문서 검색을 수행 가능
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # search_kwargs란 추가 옵션인데, 여기서 k는 반환할 문서 수를 지정해줌.
)

# 무료 api를 활용하기위해 ChatGroq 사용
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,   # 정확한 답변을 듣기 위해 temperature 낮게 설정.
    max_retries=2,     # 작업이 실패했을 때 2번정도 재시도하도록 설정
)

# context 부분과, question 부분에 입력을 받아서 한국어로 답변하는 프롬프트 생성
template = ''' Answer the question based only on the following context.

[Context]
{context}

[Question]
{question}

[Answer (in Korean)]
'''

# 위의 프롬프트를 입력 받아서 대화형 프롬프트를 생성
prompt = ChatPromptTemplate.from_template(template)

# 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
def format_docs(docs):
    return "\n\n".join(
        [doc.page_content for doc in docs]
    )

# retriever를 통해 검색한 문서를 context에 놓고
# 사용자의 질문을 받아 question 부분에 넣어 
# prompt에 넘겨주고, llm에 넘겨주고, stroutputparser로 출력하는 파이프라인 rag_chain 구성
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# su = rag_chain.invoke("미 노동부가 무엇을 발표했어?")
# print(su)

def chat(message, history):  # gradio의 채팅창에서 사용자가 질문하면 생성되는 함수를 사용함

    if not message.strip():  # 사용자가 입력을 안했을 경우 반환하는 함수를 생성함.
        return "질문을 입력해주세요."

    try:    # 에러가 발생할 가능성 있는 코드를 안전하게 실행하기 위함. (ex) api 오류, pdf 오류, ollama 연결 실패 등)
        response = rag_chain.invoke(message)  # 정상 작동됬을 경우 사용자의 질문을 rag에 넣고 답변을 생성함.
        return response   # 답변을 반환

    except Exception as e:
        return f"에러 발생:\n{str(e)}"


demo = gr.ChatInterface(
    fn=chat,        # 사용자가 질문하면 호출할 함수를 지정해줌.
    title="📚 PDF 기반 RAG 챗봇",
    description="PDF 문서를 기반으로 답변합니다.",
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(
        placeholder="질문을 입력하세요...",  # 입력창 힌트 문구를 지정한다.
        container=True,  # 입력창을 박스형태로 감싼다.
        scale=7,    # 입력창 너비 비율을 숫자 크게하면 더 넓어진다.
    ),
)


demo.launch()
