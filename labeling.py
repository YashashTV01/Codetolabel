import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from multiprocessing import Pool

sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def sentiment_analyse(sentiment_text):
    score = sid.polarity_scores(sentiment_text)
    compound_score = score['compound']
    sentiment_label = ""
    sentiment_value = 0

    if compound_score >= 0.8:
        sentiment_label = "Very Positive"
        sentiment_value = 7
    elif 0.6 <= compound_score < 0.8:
        sentiment_label = "Positive"
        sentiment_value = 6
    elif 0.4 <= compound_score < 0.6:
        sentiment_label = "Slightly Positive"
        sentiment_value = 5
    elif 0.2 <= compound_score < 0.4:
        sentiment_label = "Somewhat Positive"
        sentiment_value = 4
    elif compound_score <= -0.8:
        sentiment_label = "Very Negative"
        sentiment_value = 1
    elif -0.8 < compound_score <= -0.6:
        sentiment_label = "Negative"
        sentiment_value = 2
    elif -0.6 < compound_score <= -0.4:
        sentiment_label = "Slightly Negative"
        sentiment_value = 3
    elif -0.4 < compound_score <= -0.2:
        sentiment_label = "Somewhat Negative"
        sentiment_value = 3
    else:
        sentiment_label = "Neutral"
        sentiment_value = 4

    return sentiment_label, sentiment_value

def analyze_sentiments(df_chunk):
    df_chunk['sentiment_label'], df_chunk['sentiment_value'] = zip(*df_chunk['sentence_str'].apply(sentiment_analyse))
    return df_chunk

def process_data_chunk(df_chunk):
    df_chunk['sentence_str'] = df_chunk['sentence_str'].apply(preprocess_text)
    df_chunk = analyze_sentiments(df_chunk)
    return df_chunk

def main():
    file_path = 'philosophy_data.csv'
    chunksize = 10000  # Adjust the chunk size based on your system's memory capacity
    output_file_path = 'philosophy_data_labeled_dataset.csv'

    df_chunks = pd.read_csv(file_path, chunksize=chunksize)
    pool = Pool()  # Use multiprocessing Pool for parallel processing

    processed_chunks = pool.map(process_data_chunk, df_chunks)
    pool.close()
    pool.join()

    df_with_sentiments = pd.concat(processed_chunks)

    df_with_sentiments.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main()
