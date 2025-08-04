
import httpx
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse
import validators
from bs4 import BeautifulSoup

def validate_url_format(url):
    return validators.url(url)

def check_http_status(url, timeout=10):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            return {
                "status_code": response.status_code,
                "final_url": str(response.url),
                "error": None
            }
    except httpx.RequestError as e:
        return {
            "status_code": None,
            "final_url": None,
            "error": str(e.__class__.__name__)
        }


def check_with_browser(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=20000)
            page.wait_for_load_state("networkidle")  # Wait until full load

            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator=' ').lower().strip()

            verdict = "OK"

            # Detect CAPTCHA/bot protection
            if any(x in content for x in ["g-recaptcha", "cf-browser-verification", "hcaptcha"]):
                verdict = "CAPTCHA"

            # Detect strong signs of "Not Found" only
            elif any(kw in text for kw in [
                "page not found", "error 404", "sorry, we can't find", "404 error"
            ]):
                verdict = "Not Found"

            # Detect mostly empty pages
            elif len(text) < 50:
                verdict = "Empty Page"

            browser.close()
            return verdict

    except Exception as e:
        return f"BrowserError: {str(e)}"
