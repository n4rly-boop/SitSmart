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
url = "https://www.gettyimages.com/photos/webcam-portrait"

# Folder to save images
save_folder = "gettyimages_webcam_portraits"
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
    for _ in range(5):  # Increased to 5 scrolls for more images
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Wait for images to load after each scroll
    
    # Save page source for debugging
    with open("getty_page_source.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    logging.info("Saved page source to getty_page_source.html for debugging")
    
    # Parse page with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")
except Exception as e:
    logging.error(f"Failed to load page with Selenium: {e}")
    driver.quit()
    exit(1)

# Close Selenium
driver.quit()

# Find all image tags with the correct class
img_tags = soup.find_all("img", class_="Xc8V0Fvh0qg0lUySLpoi")
logging.info(f"Found {len(img_tags)} image tags with class 'Xc8V0Fvh0qg0lUySLpoi'.")

# Fallback: If no images are found with the specific class, try all img tags with valid src
if not img_tags:
    img_tags = [img for img in soup.find_all("img") if img.get("src") and img.get("src").startswith("https://media.gettyimages.com")]
    logging.info(f"Fallback: Found {len(img_tags)} image tags with valid Getty src URLs.")

# Download each image
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}
session = requests.Session()
for idx, img in enumerate(img_tags):
    img_url = img.get("src") or img.get("data-src") or img.get("data-lazy")
    if not img_url or not img_url.startswith("http"):
        logging.warning(f"Skipping invalid URL: {img_url}")
        continue

    try:
        img_response = session.get(img_url, headers=headers, timeout=10)
        img_response.raise_for_status()

        parsed_url = urllib.parse.urlparse(img_url)
        img_name = os.path.basename(parsed_url.path)
        if not img_name:
            img_name = f"webcam_portrait_{idx}.jpg"

        img_path = os.path.join(save_folder, img_name)
        with open(img_path, "wb") as f:
            f.write(img_response.content)
        logging.info(f"Downloaded: {img_name}")
    except Exception as e:
        logging.error(f"Failed to download {img_url}: {e}")
    time.sleep(1)  # Delay to avoid overwhelming the server

print(f"All images saved to {save_folder}")
logging.info("Image download process completed.")