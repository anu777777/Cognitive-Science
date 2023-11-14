#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install nltk


# In[2]:


import nltk


# In[3]:


nltk.download('all')


# In[4]:


nltk.download('wordnet') from nltk.stem import WordNetLemmatizerd


# In[5]:


nltk.download('wordnet')


# In[6]:


from nltk.stem import WordNetLemmatizer


# In[7]:


nltk.download('stopwords')


# In[8]:


def remove_stopwords(text):


# In[9]:


def remove_stopwords(text):
    stop_words = nltk.corpus.stopwords.words('english')
    filtered_text = [word for word in text.split() if word not in stop_words]
    return filtered_text
text = "This is a sentence with some stopwords."
filtered_text = remove_stopwords(text)
print(filtered_text)


# In[ ]:




