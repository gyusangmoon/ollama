# 찐 최종으로 학교 홈페이지 본문과 첨부파일 긁어오는 코드
import os
import re
import requests
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright


# =========================================================
# 파일명 안전 처리
# =========================================================

def safe_filename(name: str):
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name.strip() or "file"


# =========================================================
# 본문 추출
# =========================================================

def extract_content(page):

    selectors = [
        "#contents",
        ".bbs_view",
        ".view_cont",
        ".board_view",
        ".fr-view",
        "article",
        "main"
    ]

    for sel in selectors:
        try:
            el = page.locator(sel)
            if el.count() > 0:
                text = el.first.inner_text().strip()
                if len(text) > 50:
                    return text
        except:
            pass

    return ""


# =========================================================
# 첨부 다운로드
# =========================================================

def download_file(url, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()

    filename = url.split("?")[0].split("/")[-1]
    filename = safe_filename(filename)

    path = os.path.join(save_dir, filename)

    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    print("[다운로드 완료]", filename)


# =========================================================
# 크롤러 (최종 핵심)
# =========================================================

def crawl(url):

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(url, wait_until="networkidle")

        # -------------------------
        # iframe 대응
        # -------------------------
        try:
            iframe = page.locator("iframe")
            if iframe.count() > 0:
                src = iframe.first.get_attribute("src")
                if src:
                    print("[iframe 이동]", src)
                    page.goto(urljoin(url, src), wait_until="networkidle")
        except:
            pass

        # -------------------------
        # 제목
        # -------------------------
        title = page.title()

        meta = page.locator("meta[name='apple-mobile-web-app-title']")
        if meta.count() > 0:
            m = meta.get_attribute("content")
            if m:
                title = m

        # -------------------------
        # 본문
        # -------------------------
        content = extract_content(page)

        # -------------------------
        # 첨부파일
        # -------------------------
        attachments = []

        links = page.locator("a").all()

        for a in links:
            try:
                href = a.get_attribute("href")
                text = a.inner_text()

                if href and "download.do" in href:
                    attachments.append({
                        "name": text,
                        "url": urljoin(url, href)
                    })
            except:
                pass

        browser.close()

    # -------------------------
    # 저장
    # -------------------------
    save_dir = os.path.join("downloads", safe_filename(title))
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # 출력
    # -------------------------
    print("\n===== 제목 =====")
    print(title)

    print("\n===== 본문 =====")
    print(content)

    print("\n===== 첨부 =====")
    for a in attachments:
        print(a["name"], a["url"])

    # -------------------------
    # 저장 파일
    # -------------------------
    with open(os.path.join(save_dir, "본문.txt"), "w", encoding="utf-8") as f:
        f.write(title + "\n\n" + content)

    # -------------------------
    # 다운로드
    # -------------------------
    for a in attachments:
        download_file(a["url"], save_dir)


# =========================================================
# 실행
# =========================================================

if __name__ == "__main__":

    url = "https://www.kongju.ac.kr/KNU/16909/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGS05VJTJGMjEzMiUyRjQyNzU2NCUyRmFydGNsVmlldy5kbyUzRnBhZ2UlM0QxJTI2c3JjaENvbHVtbiUzRCUyNnNyY2hXcmQlM0QlMjZiYnNDbFNlcSUzRCUyNmJic09wZW5XcmRTZXElM0QlMjZyZ3NCZ25kZVN0ciUzRCUyNnJnc0VuZGRlU3RyJTNEJTI2aXNWaWV3TWluZSUzRGZhbHNlJTI2cGFzc3dvcmQlM0QlMjZDU1JGX1RPS0VOJTNEMGI5ZjA5MGItYjdjMC00MjI2LWFiNmUtYjdiOWRiM2UwODI0JTI2"

    crawl(url)