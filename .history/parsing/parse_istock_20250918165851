import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL to scrape
url = "https://www.istockphoto.com/search/2/image-film?phrase=webcam+portrait"

# Headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# Folder to save images
save_folder = "istockphoto_images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Send request to the website with a session
session = requests.Session()
try:
    response = session.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # Check for request errors
except requests.exceptions.HTTPError as e:
    logging.error(f"HTTP Error: {e}")
    print("HTTP Error occurred. Consider using Selenium for JavaScript-rendered content or check iStockphoto's terms.")
    exit(1)
except requests.exceptions.RequestException as e:
    logging.error(f"Request failed: {e}")
    exit(1)

# Parse the HTML content
soup = BeautifulSoup(response.text, "html.parser")

# Find all image tags
img_tags = soup.find_all("img")
logging.info(f"Found {len(img_tags)} image tags on the page.")

# Download each image
for idx, img in enumerate(img_tags):
    img_url = img.get("src") or img.get("data-src")  # Check for lazy-loaded images
    if not img_url or not img_url.startswith("http"):
        logging.warning(f"Skipping invalid URL: {img_url}")
        continue

    try:
        # Get image content
        img_response = session.get(img_url, headers=headers, timeout=10)
        img_response.raise_for_status()

        # Extract image filename from URL
        parsed_url = urllib.parse.urlparse(img_url)
        img_name = os.path.basename(parsed_url.path)
        if not img_name:
            img_name = f"image_{idx}.jpg"  # Fallback name

        # Save image to folder
        img_path = os.path.join(save_folder, img_name)
        with open(img_path, "wb") as f:
            f.write(img_response.content)
        logging.info(f"Downloaded: {img_name}")
    except Exception as e:
        logging.error(f"Failed to download {img_url}: {e}")
    time.sleep(1)  # Delay to avoid rate limiting

print(f"All images saved to {save_folder}")
logging.info("Image download process completed.")