#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Sample text
text = "The quick brown foxes are jumping over the lazy dogs"

# Tokenize the text
words = word_tokenize(text)

# Create a WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Apply lemmatization to each word
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

print(lemmatized_words)


# In[ ]:




