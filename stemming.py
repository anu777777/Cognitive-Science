#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Sample text
text = "The quick brown foxes are jumping over the lazy dogs"

# Tokenize the text
words = word_tokenize(text)

# Create a PorterStemmer
ps = PorterStemmer()

# Apply stemming to each word
stemmed_words = [ps.stem(word) for word in words]

print(stemmed_words)


# In[ ]:




