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
url = "https://www.istockphoto.com/search/2/image-film?phrase=webcam+portrait"

# Folder to save images
save_folder = "istockphoto_images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Set up Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Avoid bot detection
try:
    driver = webdriver.Chrome(options=chrome_options)
except Exception as e:
    logging.error(f"Failed to initialize ChromeDriver: {e}")
    print("Ensure ChromeDriver is installed and matches your Chrome version.")
    exit(1)

try:
    # Load page
    driver.get(url)
    # Wait for at least one image to be visible
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "img"))
    )
    # Scroll multiple times to load more images
    for i in range(5):  # Increased scroll attempts
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Increased wait time for lazy-loaded images
        logging.info(f"Scroll {i+1} completed")
    soup = BeautifulSoup(driver.page_source, "html.parser")
except Exception as e:
    logging.error(f"Failed to load page with Selenium: {e}")
    driver.quit()
    exit(1)

# Close Selenium
driver.quit()

# Find all image tags with 'media.istockphoto.com' in src or data-src
img_tags = soup.find_all(
    "img",
    lambda tag: (tag.get("src") and "media.istockphoto.com" in tag.get("src"))
    or (tag.get("data-src") and "media.istockphoto.com" in tag.get("data-src"))
    or (tag.get("data-lazy-src") and "media.istockphoto.com" in tag.get("data-lazy-src"))
)
logging.info(f"Found {len(img_tags)} image tags with 'media.istockphoto.com' in src or data-src.")

# Download each image
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}
session = requests.Session()
for idx, img in enumerate(img_tags):
    img_url = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
    asset_id = img.get("data-asset-id") or img.get("data-media-id")  # Check alternative attributes
    if not img_url or not img_url.startswith("http"):
        logging.warning(f"Skipping invalid URL: {img_url}")
        continue

    # Optionally filter for specific image ID (82567379)
    if asset_id == "82567379" or "82567379" in img_url:
        logging.info(f"Found target image ID: {asset_id or 'in URL'}")
    # Comment out the above condition to download all images

    try:
        img_response = session.get(img_url, headers=headers, timeout=10)
        img_response.raise_for_status()

        parsed_url = urllib.parse.urlparse(img_url)
        img_name = os.path.basename(parsed_url.path)
        if not img_name:
            img_name = f"image_{asset_id or idx}.jpg"

        img_path = os.path.join(save_folder, img_name)
        with open(img_path, "wb") as f:
            f.write(img_response.content)
        logging.info(f"Downloaded: {img_name} (Asset ID: {asset_id or 'Unknown'})")
    except Exception as e:
        logging.error(f"Failed to download {img_url}: {e}")
    time.sleep(1)  # Delay to avoid rate limiting

print(f"All images saved to {save_folder}")
logging.info("Image download process completed.")