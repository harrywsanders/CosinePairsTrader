#CosinePairsTradaer README

Hey there! This is CosinePairsTrader (I'm workshopping the name). I wanted to explore stock market dynamics and pairs-based trading using a 
text-based approach. My aim was to further my understanding of financial models and dig a little deeper into some natural language processing tech, and ahve fun along the way!

## How It Works

I start off by scraping the Fortune 500 company list from Wikipedia. I pull the company name, NYSE code, and a brief description for each of these companies. 

I then use BERT to generate some embeddings based on the description of each company. This is the idea that similar embeddings in vector space will have similar semantic significance– i.e. similar companies. 

Then, I use HDBSCAN to dynamically cluster them (this is nice because I don't have to specify groups, they arise organically). With this, I used TFIDF to validate the clustering. 

For each cluster of companies, I  use Yahoo Finance's API to download the latest stock data. I then calculate the potential return for each pair of stocks within a cluster using their most recent closing prices.

I also use an ARIMA (AutoRegressive Integrated Moving Average) model to predict the next closing price for each stock in a pair, which gives us an estimate of the future return. 

To refine the results further, I weight the predicted return by the cosine similarity of the pair's company descriptions, assuming the
similarity of two companies very loosley relates to the cosine similarity, meaning that more similar companies should more confidently be predicted as statistical anomalies. That's the project! Stay tuned for more updates.  
