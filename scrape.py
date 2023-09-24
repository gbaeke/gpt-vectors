import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_website(url, depth, base_url=None):
    if depth <= 0:
        return []

    if not base_url:
        base_url = urlparse(url).scheme + "://" + urlparse(url).netloc

    internal_links = set()
    response = requests.get(url)
    if response.status_code != 200:
        logging.warning(f"Failed to fetch URL: {url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href")
        if not href:
            continue

        full_url = urljoin(base_url, href)
        parsed_url = urlparse(full_url)

        if parsed_url.netloc == urlparse(base_url).netloc and parsed_url.scheme in ["http", "https"]:
            internal_links.add(full_url)

    scraped_links = []
    for link in internal_links:
        logging.info(f"Scraping link: {link}")
        scraped_links.extend(scrape_website(link, depth - 1, base_url))

    return list(internal_links) + scraped_links

if __name__ == "__main__":
    website_url = "https://learn.microsoft.com/en-us/azure/aks/"  # Replace with your desired website URL
    scrape_depth = 2  # Set the desired depth for scraping

    logging.info(f"Starting scraping of {website_url} to depth {scrape_depth}")

    scraped_links = scrape_website(website_url, scrape_depth)

    logging.info("Scraping completed. Internal links found:")
    for link in scraped_links:
        print(link)
