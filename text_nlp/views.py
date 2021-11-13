from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer

# Create your views here.

import pickle
import numpy as np




def process(request):
    context = {}
    
    if request.method == 'POST':
        tx = request.POST.get('text')
        txt = [tx]
        vectorizer = pickle.load(open("models/vectorizer.pickle", 'rb'))
        nb = pickle.load(open("models/classification.model", 'rb')) 
        abc_test = vectorizer.transform(txt)
        test_predict = nb.predict(abc_test)
        labels = {0: "Politics", 1: "Technology", 2:"Entertainment", 3:"Business"}
        print(test_predict)
        category = labels[test_predict[0]]
        context = {'category':category,'tx':tx}
    return render(request,'home.html',context)