# Sentiment Analysis Tool

This Python tool performs sentiment analysis on textual data using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool from the NLTK (Natural Language Toolkit) library. The tool is designed to analyze the sentiment of text data stored in a CSV file and add sentiment labels and compound scores to the dataset.

### Features:
- **Text Preprocessing:** The tool preprocesses the text data by tokenizing, lowercasing, removing punctuation, and lemmatizing the words to prepare them for sentiment analysis.
- **Sentiment Analysis:** Using the VADER sentiment analysis tool, the tool calculates the compound score for each text entry, representing the overall sentiment polarity of the text.
- **Sentiment Labeling:** Based on the compound score, the tool assigns sentiment labels to each text entry, categorizing them as VeryPositive, Positive, Neutral, Negative, or VeryNegative.
- **Parallel Processing:** To speed up the sentiment analysis process, the tool utilizes multiprocessing to analyze multiple text entries simultaneously.
- **Easy-to-Use:** The tool provides a user-friendly interface where users can input the path to their dataset, specify the column containing text data, and choose the location to save the updated dataset.

### Requirements:
- Python 3.x
- NLTK (Natural Language Toolkit) library
- Pandas library

### Usage:
1. Install the required libraries by running `pip install nltk pandas`.
2. Ensure that you have downloaded the NLTK data by running `nltk.download('punkt')` and `nltk.download('stopwords')`.
3. Prepare your dataset in CSV format with a column containing text data.
4. Run the `sentiment_analysis_tool.py` script.
5. Enter the path to your dataset when prompted.
6. Enter the name of the column containing text data.
7. Choose the location to save the updated dataset.
8. The tool will perform sentiment analysis on the text data and update the dataset with sentiment labels and compound scores.

### Example:
```bash
$ python sentiment_analysis_tool.py
Enter the path to your dataset: path/to/your/dataset.csv
Enter the name of the column containing text data: text_column
Enter the path to save the updated dataset: path/to/save/updated_dataset.csv
Sentiment analysis completed and dataset updated successfully.
```


### Acknowledgments:
- NLTK (Natural Language Toolkit) for providing the VADER sentiment analysis tool.
- Python community for developing and maintaining useful libraries.
