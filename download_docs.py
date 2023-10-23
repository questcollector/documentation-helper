import asyncio
import concurrent.futures
import logging
import pathlib
import re
import urllib

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

driver_path = pathlib.Path(__file__).parent / "chromedriver-mac-arm64/chromedriver"

# The URL to scrape
param_for_scraping = {
    "docs": {
        "url": "https://python.langchain.com/en/latest",
        "base_url": "https://python.langchain.com/en/latest",
        "condition": lambda href: href.startswith("docs/"),
        "output_dir": pathlib.Path(__file__).parent
        / "langchain-docs/python.langchain.com/en/latest",
    },
    "api": {
        "url": "https://api.python.langchain.com/en/latest/",
        "base_url": "https://api.python.langchain.com/en/latest/",
        "condition": lambda href: re.match(".+[.]html*", href),
        "output_dir": pathlib.Path(__file__).parent
        / "langchain-docs/api.python.langchain.com/en/latest",
    },
}

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 로깅 레벨 설정 (원하는 수준으로 설정)
    format="%(asctime)s [%(name)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler()],  # 로그를 stdout으로 출력
)


def fetch_url(href_url: str, count: int) -> str:
    logging.info(f"fetching document {count}: {href_url}")
    return requests.get(href_url).text


def fetch_url_with_webdriver(href_url: str, count: int) -> str:
    logging.info(f"fetching document {count}: {href_url}")
    with webdriver.Chrome(service=Service(ChromeDriverManager().install())) as driver:
        try:
            driver.get(href_url)
            return driver.page_source
        except:
            logging.info(f"error on fetching document {count}: {href_url}")
            return href_url


def save_file(file_response: str, file_name: pathlib.Path, count: int):
    logging.info(f"saving document {count}: {file_name.as_posix()}")
    file_name.parent.mkdir(exist_ok=True, parents=True)
    file_name.write_text(file_response, encoding="utf-8")


async def scrape_documents(href: str, site: str, count: int):
    logging.info(f"\tdocument {count} in {site}: {href}")

    url = param_for_scraping[site]["base_url"]
    condition = param_for_scraping[site]["condition"]
    output_dir = param_for_scraping[site]["output_dir"]

    href = re.sub("^/", "", href)

    if condition(href):
        loop = asyncio.get_event_loop()

        # Make a full URL if necessary
        if not href.startswith("http"):
            href_url = urllib.parse.urljoin(url, href)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Fetch the .html file
            await asyncio.sleep(0.1)
            # file_response = await loop.run_in_executor(None, fetch_url, href_url, count)
            file_response = await loop.run_in_executor(
                None, fetch_url_with_webdriver, href_url, count
            )

            # Write it to a file
            await asyncio.sleep(0.1)

            if not href.endswith(".html"):
                href = href + ".html"

            file_name = output_dir / href
            await loop.run_in_executor(
                executor, save_file, file_response, file_name, count
            )


async def scrape_langchain(site: str):
    url = param_for_scraping[site]["url"]

    logging.info(f"main site of {site}: {url}")

    # Fetch the page
    with webdriver.Chrome(service=Service(ChromeDriverManager().install())) as driver:
        driver.get(url)
        response = driver.page_source
    soup = BeautifulSoup(response, "html.parser")
    # Find all links to documents
    links = soup.find_all("a", href=True)
    hrefs = set(map(lambda link: re.sub("#.+", "", link["href"]), links))

    tasks = [
        asyncio.ensure_future(scrape_documents(href, site, i))
        for i, href in enumerate(hrefs)
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(scrape_langchain("docs"))
    asyncio.run(scrape_langchain("api"))
