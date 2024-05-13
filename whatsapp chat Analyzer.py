import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load WhatsApp chat data
data = pd.read_csv("whatsapp_chat.csv")

# Preprocessing (if needed)
# Clean and preprocess the chat data

# Sentiment Analysis
data['Sentiment'] = data['Message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Word Frequency Analysis
word_freq = data['Message'].str.split(expand=True).stack().value_counts()

# Visualization
# Plot sentiment distribution
plt.hist(data['Sentiment'], bins=20, color='skyblue', edgecolor='black')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

# Plot word frequency
word_freq[:20].plot(kind='bar', color='orange')
plt.title('Top 20 Word Frequencies')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.show()
