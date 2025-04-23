
# New York Airbnb Data Analytics and Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
mydata = pd.read_csv('AB_NYC_2019.csv')

# Preprocessing
mydata["last_review"] = pd.to_datetime(mydata["last_review"])
mydata['reviews_per_month'] = mydata['reviews_per_month'].fillna(mydata['reviews_per_month'].mean())
mydata.drop(columns=['host_name', 'last_review'], axis=1, inplace=True)

# Text cleaning
mydata['name'].fillna('', inplace=True)
def clean_text(line):
    line = re.sub('[^A-Za-z]+', ' ', line).lower()
    tokens = nltk.word_tokenize(line)
    return " ".join([w for w in tokens if w not in stopwords.words('english')])

mydata['final_name'] = mydata['name'].apply(clean_text)

# Classification: Cheap vs Expensive
mydata['target'] = mydata['price'].apply(lambda x: 1 if x > 300 else 0)
train, test = train_test_split(mydata, test_size=0.2, random_state=315, stratify=mydata['target'])
vect = TfidfVectorizer()
X_train = vect.fit_transform(train['final_name'])
X_test = vect.transform(test['final_name'])
y_train = train['target']
y_test = test['target']

ros = RandomOverSampler(sampling_strategy='minority', random_state=1)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

model = LGBMClassifier(random_state=315)
model.fit(X_train_ros, y_train_ros)
preds = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, preds))
print("Accuracy:", accuracy_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))

# Regression Model
mydata.drop(columns=['target'], inplace=True)
mydata = mydata[mydata.price > 0]
mydata = mydata[mydata.availability_365 > 0]

label_columns = ['neighbourhood_group', 'neighbourhood', 'room_type']
for col in label_columns:
    le = LabelEncoder()
    mydata[col] = le.fit_transform(mydata[col])

X = mydata[['neighbourhood_group', 'neighbourhood', 'room_type', 'minimum_nights',
            'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
            'availability_365']]
y = np.log10(mydata['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("Regression Metrics:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
