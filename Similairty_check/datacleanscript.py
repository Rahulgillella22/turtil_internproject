import json
import re
import unicodedata

# Function to clean and normalize text
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)  # Normalize special unicode characters
    text = text.replace('\u2013', '-')  # Replace en dash with normal dash
    text = re.sub(r'[^\w\s\-`\'=./]', '', text)  # Allow =, ., /, etc.
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    return text.strip().lower()  # Final cleanup and lowercase

# Load the JSON file
with open('merged_all_posts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Clean title and body
for post in data:
    post['post_title'] = clean_text(post['post_title'])
    post['post_body'] = clean_text(post['post_body'])

# Optional: Save cleaned data to a new file
with open('cleaned_posts.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("✅ Cleaning completed. Check 'cleaned_posts.json'")
