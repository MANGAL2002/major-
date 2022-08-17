import numpy as np
from flask import Flask, request, jsonify, render_template           # render_template it is used to redirect
#from flask_ngrok import run_with_ngrok

import pickle
from PIL import Image
#import streamlit as st 
import matplotlib.pyplot as plt
import pandas as pd
#st.set_option('deprecation.showfileUploaderEncoding', False)


app = Flask(__name__)
model = pickle.load(open('NLP_model_naivebased.pkl','rb')) 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
print(cv)
corpus=pd.read_csv('corpus_dataset.csv')
corpus1=corpus['tweets'].tolist()
X = cv.fit_transform(corpus1).toarray()

#for globaly ngrok used
#run_with_ngrok(app)

@app.route('/')     # it is a deacouterator
def home():
  
    return render_template("index.html")
@app.route('/')     # it is a deacouterator
def home():
  
    return render_template("aboutus.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    tweets = request.args.get('tweets')
    
    tweets=[tweets]

    input_data = cv.transform(tweets).toarray()

    prediction = model.predict(input_data)

    #input_pred = input_pred.astype(int)
    
    if prediction[0]==2:
      return render_template('index.html', prediction_text='Tweets is Positive')
      
    elif prediction[0]==1:    
      return render_template('index.html', prediction_text='Tweets is Negative')
    else:
      return render_template('index.html', prediction_text='Tweets is Netural')
      



if __name__ == "__main__":
  app.run(debug=True)
    
    
