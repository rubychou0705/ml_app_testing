# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import joblib
import autoai_libs
import numpy as np

'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
'''

model = joblib.load('P4.pickle')