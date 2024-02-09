import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    compound_score = score['compound']

    if compound_score >= 0.8:
        return "Very Positive"
    elif 0.6 <= compound_score < 0.8:
        return "Positive"
    elif 0.4 <= compound_score < 0.6:
        return "Slightly Positive"
    elif 0.2 <= compound_score < 0.4:
        return "Somewhat Positive"
    elif compound_score <= -0.8:
        return "Very Negative"
    elif -0.8 < compound_score <= -0.6:
        return "Negative"
    elif -0.6 < compound_score <= -0.4:
        return "Slightly Negative"
    elif -0.4 < compound_score <= -0.2:
        return "Somewhat Negative"
    else:
        return "Neutral"


def analyze_sentiments(df, input_column):
    df['sentiment'] = df[input_column].apply(sentiment_analyse)
    return df

file_path = 'Shakespeare_data.csv'
df = pd.read_csv(file_path)

selected_column = 'PlayerLine'

df_with_sentiments = analyze_sentiments(df, selected_column)

output_file_path = 'labeled_dataset.csv'
df_with_sentiments.to_csv(output_file_path, index=False)
