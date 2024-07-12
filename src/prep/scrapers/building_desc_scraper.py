import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Function to scrape article links from a page
def get_article_links(page_url):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_links = []
    
    for article in soup.find_all('article'):
        link = article.find('a', href=True)
        if link and "eventsguide" not in link['href'] and "dezeenjobs" not in link['href']:
            article_links.append(link['href'])
            
    return article_links

# Function to scrape the title and main text from an article
def scrape_article_details(article_url):
    response = requests.get(article_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract title
    title_tag = soup.find('a', href=article_url)
    title = title_tag.text if title_tag else 'No title found'
    
    # Extract article text
    article_text = ''
    article_section = soup.find('article')
    if article_section:
        for paragraph in article_section.find_all('p'):
            article_text += paragraph.text + ' '
    
    return title, article_text.strip()

# Main function to create the training set
def create_training_set(main_page_url, num_articles):
    page = 1
    articles = []
    article_count = 0
    
    while article_count < num_articles:
        page_url = f"{main_page_url}page/{page}/"
        article_links = get_article_links(page_url)
        
        for link in article_links:
            if article_count >= num_articles:
                break
            title, text = scrape_article_details(link)
            articles.append({'title': title, 'article': text})
            article_count += 1
        
        page += 1
    
    return articles

# URL of the Dezeen architecture page
main_page_url = 'https://www.dezeen.com/architecture/'

# Number of articles to scrape
num_articles = 15000  # Adjust as needed

# Create the training set
training_set = create_training_set(main_page_url, num_articles)

# Save the training set to a CSV file
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, '../../../data/raw/misc')

df = pd.DataFrame(training_set)
df.to_csv(os.path.join(save_path, 'building_descriptions.csv'), index=False)

print('Training set created and saved to ../../../data/raw/misc/building_descriptions.csv')
