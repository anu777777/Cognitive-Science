#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample text
text = "The quick brown foxes are jumping over the lazy dogs"

# Tokenize the text
words = word_tokenize(text)

# Perform POS tagging
pos_tags = nltk.pos_tag(words)

print(pos_tags)


# In[ ]:




