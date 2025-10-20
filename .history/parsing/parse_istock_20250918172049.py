from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
import urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import time
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL to scrape
url = "https://www.istockphoto.com/search/2/image-film?phrase=webcam+portrait"

# Folder to save images
save_folder = "istockphoto_images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Path to ChromeDriver (update this with your actual path)
chromedriver_path = "/path/to/chromedriver"  # e.g., "/usr/local/bin/chromedriver"

# Set up Selenium
chrome_options = Options()
chrome_options.add_argument("--headless=new")  # Use new headless mode
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Avoid bot detection
chrome_options.add_argument("--no-sandbox")  # For compatibility
chrome_options.add_argument("--disable-dev-shm-usage")  # Prevent memory issues
chrome_options.add_argument("--log-level=3")  # Reduce Chrome logging noise

# Enable ChromeDriver logging for debugging
service = Service(executable_path=chromedriver_path, log_path="chromedriver.log")

try:
    driver = webdriver.Chrome(service=service, options=chrome_options)
    logging.info("ChromeDriver initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize ChromeDriver: {e}")
    print("Ensure ChromeDriver is installed, executable, and matches your Chrome version.")
    print(f"Check chromedriver.log for details.")
    exit(1)

try:
    # Load page
    driver.get(url)
    logging.info(f"Loaded URL: {url}")
    # Wait for initial images to load
    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img[src*='media.istockphoto.com']"))
    )

    # Handle cookie popup if present
    try:
        cookie_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'Agree ')]"))
        )
        cookie_button.click()
        logging.info("Clicked cookie accept button")
        time.sleep(1)
    except:
        logging.info("No cookie popup found or not clickable")

    # Dynamic scrolling to load all images
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0
    max_attempts = 10  # Prevent infinite loops
    while scroll_attempts < max_attempts:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for new content
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            logging.info("No more content to load")
            break
        last_height = new_height
        scroll_attempts += 1
        logging.info(f"Scroll attempt {scroll_attempts} completed")

    # Save page source for debugging
    with open("page_source.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    logging.info("Saved page source to page_source.html")

    # Parse page with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")
except Exception as e:
    logging.error(f"Failed to load or process page with Selenium: {e}")
    driver.quit()
    exit(1)
finally:
    driver.quit()
    logging.info("Selenium driver closed")

# Find all image tags
img_tags = soup.find_all(
    "img",
    lambda tag: tag.get("src", "").startswith("https://media.istockphoto.com") or
                tag.get("data-src", "").startswith("https://media.istockphoto.com") or
                tag.get("data-lazy-src", "").startswith("https://media.istockphoto.com")
)
logging.info(f"Found {len(img_tags)} image tags matching criteria.")

# Set up requests session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}

# Download each image
for idx, img in enumerate(img_tags):
    img_url = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
    asset_id = img.get("data-asset-id") or img.get("data-media-id") or hashlib.md5(img_url.encode()).hexdigest()[:8]

    if not img_url or not img_url.startswith("https://media.istockphoto.com"):
        logging.warning(f"Skipping invalid URL: {img_url}")
        continue

    try:
        img_response = session.get(img_url, headers=headers, timeout=10)
        img_response.raise_for_status()

        parsed_url = urllib.parse.urlparse(img_url)
        img_name = os.path.basename(parsed_url.path) or f"image_{asset_id}.jpg"
        img_path = os.path.join(save_folder, img_name)

        # Avoid overwriting existing files
        base, ext = os.path.splitext(img_name)
        counter = 1
        while os.path.exists(img_path):
            img_path = os.path.join(save_folder, f"{base}_{counter}{ext}")
            counter += 1

        with open(img_path, "wb") as f:
            f.write(img_response.content)
        logging.info(f"Downloaded: {img_path} (Asset ID: {asset_id})")
    except Exception as e:
        logging.error(f"Failed to download {img_url}: {e}")
    time.sleep(0.5)  # Polite delay

# Clean up
session.close()
print(f"All images saved to {save_folder}")
logging.info("Image download process completed.")