import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan

url = 'https://en.wikipedia.org/wiki/List_of_Fortune_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table', {'class': 'wikitable sortable'})

rows = table.find_all('tr')
companies = []
for row in rows[1:]:
    data = row.find_all('td')
    company_name = data[1].text.strip()
    companies.append(company_name)

def get_company_info(company_name):
    url = f'https://en.wikipedia.org/wiki/{company_name}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    infobox = soup.find('table', {'class': 'infobox vcard'})
    if infobox is None:
        return None, None
    nyse_code = None
    for row in infobox.find_all('tr'):
        if 'NYSE' in row.text:
            nyse_code = row.find('td').text.strip()
            break

    description = soup.find('p').text.strip()

    return nyse_code, description

company_info = []
for company in companies:
    nyse_code, description = get_company_info(company)
    company_info.append((company, nyse_code, description))

df = pd.DataFrame(company_info, columns=['Company', 'NYSE Code', 'Description'])

df = df.dropna(subset=['Description'])

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(tfidf_matrix.toarray())

df['Cluster'] = cluster_labels

df.to_csv('clustered_companies.csv', index=False)
