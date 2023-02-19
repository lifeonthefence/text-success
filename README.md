# text-success

This project aims to identify which features of Tweets are most important for its success, as measured by the number of Retweets.

Approach taken:
- Snscrape will be used to gather texts from a given source, along with their metrics of success and other variables.
Natural language processing will be used to clean the dataset and extract features from the texts such as complexity/length, topic, sentiment, and key words.
Based on these features, a model will be trained to assess the relative importance of these in creating ‘successful’ texts; regression or more advanced (but interpretable) models can be used.
