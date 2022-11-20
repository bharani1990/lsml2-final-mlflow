from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import tree
import nltk.data

nltk.download('stopwords')
from nltk.corpus import stopwords

print('Running loaders...')

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
tokenizer = TweetTokenizer()
stop = stopwords.words('english')
lr_clf = LogisticRegression(random_state=42)
sgd_clf = SGDClassifier(loss='log_loss', random_state=42)
rf_clf = RandomForestClassifier(n_estimators=20, random_state=42)
xgb_clf = XGBClassifier(random_state=42)
tree_clf = tree.DecisionTreeClassifier(random_state=42)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None]}]
print('Running loaders completed...')
