from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
import urllib.parse
import requests
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL to scrape
url = "https://stock.adobe.com/search/images?k=webcam%20portrait"

# Folder to save images
save_folder = "adobe_stock_images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Set up Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")
driver = webdriver.Chrome(options=chrome_options)

try:
    # Load page
    driver.get(url)
    
    # Wait for at least one image to be present
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "img"))
    )
    
    # Scroll multiple times to load all images
    for _ in range(3):  # Adjust number of scrolls as needed
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Wait for images to load after each scroll
    
    # Save page source for debugging
    with open("page_source.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    logging.info("Saved page source to page_source.html for debugging")
    
    # Parse page with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")
except Exception as e:
    logging.error(f"Failed to load page with Selenium: {e}")
    driver.quit()
    exit(1)

# Close Selenium
driver.quit()

# Find all image tags (more general selector)
img_tags = soup.find_all("img")
logging.info(f"Found {len(img_tags)} image tags on the page.")

# Download each image
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}
session = requests.Session()
for idx, img in enumerate(img_tags):
    img_url = img.get("src") or img.get("data-src")
    if not img_url or not img_url.startswith("http"):
        logging.warning(f"Skipping invalid URL: {img_url}")
        continue

    try:
        img_response = session.get(img_url, headers=headers, timeout=10)
        img_response.raise_for_status()

        parsed_url = urllib.parse.urlparse(img_url)
        img_name = os.path.basename(parsed_url.path)
        if not img_name:
            img_name = f"image_{idx}.jpg"

        img_path = os.path.join(save_folder, img_name)
        with open(img_path, "wb") as f:
            f.write(img_response.content)
        logging.info(f"Downloaded: {img_name}")
    except Exception as e:
        logging.error(f"Failed to download {img_url}: {e}")
    time.sleep(1)

print(f"All images saved to {save_folder}")
logging.info("Image download process completed.")