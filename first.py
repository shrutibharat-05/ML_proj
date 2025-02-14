import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# Load your data
data = pd.read_csv("C:\\Users\\shrut\\Downloads\\1news_yess.csv", index_col=0)

# Display the first few rows
print(data)  # This will show the data in the terminal

print(data.isnull().sum())

sns.countplot(data=data,
              x='Label',
              order=data['Label'].value_counts().index)
plt.show()

def preprocess_text(text_data):
    preprocessed_text = []
    
    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower()
                                  for token in str(sentence).split()
                                  if token not in stopwords.words('english')))

    return preprocessed_text

# FAKE News WordCloud
fake_text = ' '.join(data['Content'][data['Label'] =="FAKE"].dropna().astype(str))
real_text = ' '.join(data['Content'][data['Label'] == "REAL"].dropna().astype(str))

print(f"Length of FAKE text: {len(fake_text)}")
print(f"Length of REAL text: {len(real_text)}")

# Generate WordClouds
if len(fake_text) > 0:
    fake_wc = WordCloud(width=1600, height=800, background_color='white').generate(fake_text)
    plt.figure(figsize=(15, 10))
    plt.imshow(fake_wc, interpolation='bilinear')
    plt.title("FAKE News Word Cloud")
    plt.axis('off')
    plt.show()

if len(real_text) > 0:
    real_wc = WordCloud(width=1600, height=800, background_color='white').generate(real_text)
    plt.figure(figsize=(15, 10))
    plt.imshow(real_wc, interpolation='bilinear')
    plt.title("REAL News Word Cloud")
    plt.axis('off')
    plt.show()


from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd

# Function to get top N words from a given corpus
def get_top_n_words(corpus, n=None):
    corpus = [doc for doc in corpus if len(doc.strip()) > 0]  # Remove empty documents
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Function to plot top words
def plot_top_words(words, title, color):
    df = pd.DataFrame(words, columns=['Word', 'Count'])
    plt.figure(figsize=(12, 6))
    df.groupby('Word').sum()['Count'].sort_values(ascending=False).plot(
        kind='bar', color=color, title=title
    )
    plt.xlabel('Top Words')
    plt.ylabel('Count')
    plt.show()

# Get top 20 words for FAKE news (Label == 0)
fake_corpus = data['Content'][data['Label'] == 'FAKE'].dropna().astype(str).tolist()
fake_words = get_top_n_words(fake_corpus, 20)
plot_top_words(fake_words, "Top 20 Words in FAKE News", "salmon")

# Get top 20 words for REAL news (Label == 1)
real_corpus = data['Content'][data['Label'] == 'REAL'].dropna().astype(str).tolist()
real_words = get_top_n_words(real_corpus, 20)
plot_top_words(real_words, "Top 20 Words in REAL News", "seagreen")


#Converting text into Vectors
x_train, x_test, y_train, y_test = train_test_split(data['Content'], 
                                                    data['Label'], 
                                                    test_size=0.25)

  
vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)                                                  

# Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)

# testing the model
print(accuracy_score(y_train, model.predict(x_train)))
print(accuracy_score(y_test, model.predict(x_test)))


#DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# testing the model
print(accuracy_score(y_train, model.predict(x_train)))
print(accuracy_score(y_test, model.predict(x_test)))


#The confusion matrix for Decision Tree Classifier 
cm = metrics.confusion_matrix(y_test, model.predict(x_test))

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                            display_labels=["FAKE", "REAL"])

cm_display.plot(cmap='Blues')  # Optional: Use a color map for better visualization
plt.title("Confusion Matrix")
plt.show()