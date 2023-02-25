# Tweets Collection Update Log

## 1.0.0
Try to collect tweets by keyword 'Ukraine' in 2022 and filter out tweets with `view_count = tweet.likeCount + tweet.retweetCount * 5 >= 100`.
Store the collected data in Tweets_ukraine_100plus.csv

## 1.0.1
Also added a search for `#UkraineRussianWar` and narrowed the collection down to 2000 to speed up collection
### problem
After add the `#UkraineRussianWar`to the search, it becomes very slow.

## 1.1
Final version of collecting tweets
Try to spread them evenly over each month, But the problem still exists. The most of the tweets on from the end of each month.
I also tried to avoid collecting duplicate tweets, but I haven't been successful so far. If you guys find that there is too much repetition when processing data, maybe I can further expand the amount of data collected to mitigate the impact of this part.
