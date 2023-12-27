#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[3]:


def relu(x):
    return np.maximum(0, x)


# In[4]:


def tanh(x):
    return np.tanh(x)


# In[ ]:




