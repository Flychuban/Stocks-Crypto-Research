import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests

model_name = "human-centered-summarization/financial-summarization-pegasus"
summarization_tokenizer = PegasusTokenizer.from_pretrained(model_name)
summarization_model = PegasusForConditionalGeneration.from_pretrained(model_name)


MODEL_NAME = 'RashidNLP/Finance_Multi_Sentiment'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 3).to(device)
sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


monitored_tickers = ['BTC', 'TSLA', 'ETH']

def search_stock_news_urls(ticker):
  search_url = f'https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws'
  r = requests.get(search_url)
  soup = BeautifulSoup(r.text, 'html.parser')
  atags = soup.find_all('a')
  hrefs = [link['href'] for link in atags]
  return hrefs

raw_urls = {ticker: search_stock_news_urls(ticker) for ticker in monitored_tickers}


exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']

def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls: 
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}
print(f"Cleaned URLs: {cleaned_urls}")

def scrape_and_process(URLs):
  ARTICLES = []
  for url in URLs:
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = [paragraph.text for paragraph in paragraphs]
    words = ' '.join(text).split(' ')[:350]
    ARTICLE = ' '.join(words)
    ARTICLES.append(ARTICLE)
  return ARTICLES

articles = {ticker: scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}


def summarize(articles):
  summaries = []
  for article in articles:
    input_ids = summarization_tokenizer.encode(article, return_tensors = 'pt')
    output = summarization_model.generate(input_ids, max_length = 55, num_beams = 5, early_stopping = True)
    summary = summarization_tokenizer.decode(output[0], skip_special_tokens=True)
    summaries.append(summary)
  return summaries

summaries = {ticker: summarize(articles[ticker]) for ticker in monitored_tickers}
print(f"Summaries: {summaries}")


def get_sentiment(sentences):
    output = []
    bert_dict = {}
    vectors = sentiment_tokenizer(sentences, padding = True, max_length = 65, return_tensors='pt').to(device)
    outputs = bert_model(**vectors).logits
    probs = torch.nn.functional.softmax(outputs, dim = 1)
    for prob in probs:
        bert_dict['neg'] = round(prob[0].item(), 3)
        bert_dict['neu'] = round(prob[1].item(), 3)
        bert_dict['pos'] = round(prob[2].item(), 3)
        results = []
        results.append(bert_dict['neg'])
        results.append(bert_dict['neu'])
        results.append(bert_dict['pos'])    
        max_value = results[0]
        label = 'NEGATIVE'
        if results[1] > max_value and results[1] > results[2]:
          max_value = results[1]
          label = 'NEUTRAL'
        elif results[1] > max_value and not results[1] > results[2]:
          max_value = results[2]
          label = 'POSITIVE'
        elif not results[1] > max_value and results[2] > max_value:
          max_value = results[2]
          label = 'POSITIVE'
        
        output.append([label, max_value])
    return output

scores = {ticker: get_sentiment(summaries[ticker]) for ticker in monitored_tickers}
print(f"Scores: {scores}")

def create_output_array(summaries, scores, cleaned_urls):
  output = []
  for ticker in monitored_tickers:
    for counter in range(len(summaries[ticker])):
      output_this = [
          ticker,
          summaries[ticker][counter],
          scores[ticker][counter][0], # this is for LABEL
          scores[ticker][counter][1],  # this is for PERCENTAGE
          cleaned_urls[ticker][counter]
      ]
      output.append(output_this)
  return output

final_output = create_output_array(summaries, scores, cleaned_urls)

final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Score', 'URL'])
print("Final Output Created!")

def upload_results():
  with open("summary.csv", mode='w', newline='') as file:
    csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)
    print("File uploaded successfully")

upload_results()