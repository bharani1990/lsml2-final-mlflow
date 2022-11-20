import pandas as pd
from nltk.tokenize import TweetTokenizer
import re
from sklearn.model_selection import train_test_split

print('Loading data...')
train_val = pd.read_csv('train_val_tweets.csv')
print('Loading data completed...')
texts = train_val['tweet'].values
target = train_val['label'].values
texts_train, texts_test, y_train, y_test = train_test_split(texts, target, test_size=0.2, random_state=42)


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    tokenizer = TweetTokenizer()
    text = ' '.join(tokenizer.tokenize(text.lower()))
    return text


texts_train = [preprocessor(t) for t in texts_train]
texts_test = [preprocessor(t) for t in texts_test]
