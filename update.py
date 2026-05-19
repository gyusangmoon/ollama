# PDF Loader 비교
# 텍스트 위주 문서 : PyMuPDFLoader, PyPDFLoader -> 텍스트만 빠르게 추출하는데 용이 but, 이미지나 표는 무시, 원본 레이아웃 보존 못함
# 표 많은 문서 : PDFPlumberLoader -> 표 추출 기능이 강점
# 레이아웃 분석 필요 : UnstructuredPDFLoader -> 텍스트, 이미지, 표 추출 및 구조 보존 능력이 뛰어남 but, 속도가 느림
# 스캔본/이미지 : PyMuPDFLoader + Tesseract OCR -> 높은 정확도로 텍스트 및 이미지 추출 지원

'''
pip install unstructured unstructured-inference
'''

from langchain_community.document_loaders import UnstructuredPDFLoader

pdf_filepath = "./AI브리프_3월_260303.pdf"