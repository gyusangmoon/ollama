import os
import re
import asyncio
from urllib.parse import urljoin, urlparse

import requests
from playwright.async_api import async_playwright

# =========================================
# 대상 URL
# =========================================
TARGET_URL = "https://www.kongju.ac.kr/KNU/16909/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGS05VJTJGMjEzMiUyRjQyNzU2NCUyRmFydGNsVmlldy5kbyUzRnBhZ2UlM0QxJTI2c3JjaENvbHVtbiUzRCUyNnNyY2hXcmQlM0QlMjZiYnNDbFNlcSUzRCUyNmJic09wZW5XcmRTZXElM0QlMjZyZ3NCZ25kZVN0ciUzRCUyNnJnc0VuZGRlU3RyJTNEJTI2aXNWaWV3TWluZSUzRGZhbHNlJTI2cGFzc3dvcmQlM0QlMjZDU1JGX1RPS0VOJTNEMGI5ZjA5MGItYjdjMC00MjI2LWFiNmUtYjdiOWRiM2UwODI0JTI2"

# =========================================
# 저장 폴더
# =========================================
SAVE_DIR = "knu_post"
ATTACH_DIR = os.path.join(SAVE_DIR, "attachments")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ATTACH_DIR, exist_ok=True)

# =========================================
# requests 세션
# =========================================
session = requests.Session()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


# =========================================
# 유틸
# =========================================
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)[:100]


# =========================================
# 첨부파일 다운로드
# =========================================
def download_file(url):
    try:
        filename = os.path.basename(urlparse(url).path)

        if not filename:
            filename = "downloaded_file"

        filepath = os.path.join(ATTACH_DIR, filename)

        r = session.get(url, headers=HEADERS, stream=True, timeout=30)
        r.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        print(f"[다운로드 완료] {filename}")

    except Exception as e:
        print(f"[다운로드 실패] {url}")
        print(e)


# =========================================
# 본문 추출
# =========================================
async def extract_content(page):

    selectors = [
        ".board_view",
        ".board-view",
        ".board_view_content",
        ".view-content",
        ".view_detail",
        ".tb_contents",
        ".dbData",
        "#contents",
        ".artclView",
    ]

    # 우선 selector 기반 시도
    for sel in selectors:
        try:
            el = await page.query_selector(sel)

            if el:
                text = await el.inner_text()
                text = clean(text)

                if len(text) > 30:
                    return text

        except:
            pass

    # fallback → body 전체
    body = await page.query_selector("body")

    if body:
        text = await body.inner_text()
        text = clean(text)

        # 노이즈 제거
        noise_words = [
            "로그인",
            "사이트맵",
            "메뉴",
            "검색",
            "공주대학교",
            "KNU",
        ]

        for n in noise_words:
            text = text.replace(n, "")

        return text

    return ""


# =========================================
# 제목 추출
# =========================================
async def extract_title(page):

    selectors = [
        "h2",
        ".board_view_tit",
        ".title",
        ".tit",
    ]

    for sel in selectors:
        try:
            el = await page.query_selector(sel)

            if el:
                text = clean(await el.inner_text())

                if text:
                    return text

        except:
            pass

    return "No Title"


# =========================================
# 첨부파일 추출
# =========================================
async def extract_attachments(page):

    files = set()

    anchors = await page.query_selector_all("a[href]")

    for a in anchors:

        try:
            href = await a.get_attribute("href")

            if not href:
                continue

            # 파일 확장자 기준
            if any(
                ext in href.lower()
                for ext in [
                    ".hwp",
                    ".hwpx",
                    ".pdf",
                    ".xls",
                    ".xlsx",
                    ".doc",
                    ".docx",
                    ".zip",
                    ".png",
                    ".jpg",
                ]
            ):
                full_url = urljoin(page.url, href)
                files.add(full_url)

            # download.do 패턴 대응
            elif "download" in href.lower():
                full_url = urljoin(page.url, href)
                files.add(full_url)

        except:
            pass

    return list(files)


# =========================================
# 메인 크롤링
# =========================================
async def crawl():

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=True
        )

        page = await browser.new_page()

        print("[접속]")
        print(TARGET_URL)

        # 페이지 이동
        await page.goto(
            TARGET_URL,
            wait_until="domcontentloaded",
            timeout=60000
        )

        # JS 렌더링 대기
        await page.wait_for_timeout(3000)

        # 제목
        title = await extract_title(page)

        # 본문
        content = await extract_content(page)

        # 첨부파일
        attachments = await extract_attachments(page)

        # 출력 확인
        print("\n[제목]")
        print(title)

        print("\n[본문 일부]")
        print(content[:1000])

        print("\n[첨부파일]")
        for f in attachments:
            print(f)

        # txt 저장
        filename = safe_filename(title)

        txt_path = os.path.join(
            SAVE_DIR,
            f"{filename}.txt"
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(title + "\n\n")
            f.write(content + "\n\n")
            f.write("첨부파일\n")
            f.write("=" * 50 + "\n")

            for file_url in attachments:
                f.write(file_url + "\n")

        print(f"\n[저장 완료] {txt_path}")

        # 첨부파일 다운로드
        for file_url in attachments:
            download_file(file_url)

        await browser.close()


# =========================================
# 실행
# =========================================
if __name__ == "__main__":
    asyncio.run(crawl())