import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from multiprocessing import Pool, cpu_count

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token not in string.punctuation]
    return ' '.join(tokens)

def perform_sentiment_analysis(text):
    preprocessed_text = preprocess_text(text)
    compound_score = analyzer.polarity_scores(preprocessed_text)['compound']
    if compound_score >= 0.6:
        return "VeryPositive", compound_score
    elif 0.2 <= compound_score < 0.6:
        return "Positive", compound_score
    elif -0.2 <= compound_score < 0.2:
        return "Neutral", compound_score
    elif -0.6 <= compound_score < -0.2:
        return "Negative", compound_score
    else:
        return "VeryNegative", compound_score

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("File not found.")
        return None

def main():
    file_path = input("Enter the path to your dataset: ")
    df = load_dataset(file_path)
    if df is None:
        return
    
    text_column = input("Enter the name of the column containing text data: ")
    
    num_processes = cpu_count()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(perform_sentiment_analysis, df[text_column])
    
    sentiment_labels, compound_scores = zip(*results)
    
    df['SentimentLabel'] = sentiment_labels
    df['CompoundScore'] = compound_scores
    
    output_path = input("Enter the path to save the updated dataset: ")
    df.to_csv(output_path, index=False)
    print("Sentiment analysis completed and dataset updated successfully.")

if __name__ == "__main__":
    main()
