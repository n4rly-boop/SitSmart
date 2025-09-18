import requests
from bs4 import BeautifulSoup
import os
import urllib.parse

# URL to scrape
url = "https://www.freepik.com/free-photos-vectors/webcam-portrait"

# Headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}

# Folder to save images
save_folder = "freepik_images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Send request to the website
response = requests.get(url, headers=headers)
response.raise_for_status()  # Check for request errors

# Parse the HTML content
soup = BeautifulSoup(response.text, "html.parser")

# Find all image tags
img_tags = soup.find_all("img")

# Download each image
for idx, img in enumerate(img_tags):
    img_url = img.get("src")
    if not img_url or not img_url.startswith("http"):
        continue  # Skip invalid or relative URLs

    try:
        # Get image content
        img_response = requests.get(img_url, headers=headers)
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
        print(f"Downloaded: {img_name}")
    except Exception as e:
        print(f"Failed to download {img_url}: {e}")

print(f"All images saved to {save_folder}")