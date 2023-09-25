from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import os
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns

main = tkinter.Tk()
main.title("AUTOMATING E-GOVERNMENT SERVICES WITH MACHINE LEARNING AND ARTIFICIAL INTELLIGENCE")
main.geometry("1300x1200")

global dataset_file
global train_features, label_features
global machine_learning
global data
global X_train, X_test, y_train, y_test


def loadData():
    global dataset_file
    global data
    dataset_file = filedialog.askopenfilename(initialdir="insurance_dataset")
    pathlabel.config(text=dataset_file)
    text.delete('1.0', END)
    text.insert(END,dataset_file+" insurance dataset loaded\n\n")
    data = pd.read_csv(dataset_file)
    text.insert(END,str(data.head()))
    
                        

def exploreData():
    global data
    global train_features, label_features
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)

    data['sex'] = data['sex'].apply({'male':0,      'female':1}.get) 
    data['smoker'] = data['smoker'].apply({'yes':1, 'no':0}.get)
    data['region'] = data['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)
    text.insert(END,str(data.head())+"\n\n")

    data1 = data.values
    train_features = data1[:,0:data.shape[1]-1]
    label_features = data1[:,data.shape[1]-1]
    X_train, X_test, y_train, y_test = train_test_split(train_features, label_features, test_size = 0.2, random_state = 0)

    text.insert(END,"Total records in dataset : "+str(len(train_features))+"\n")
    text.insert(END,"80% dataset records used to train ML : "+str(len(X_train))+"\n")
    text.insert(END,"20% dataset records used to test ML Accuracy : "+str(len(X_test))+"\n")

    sns.jointplot(x=data['age'],y=data['charges'])
    plt.show()

def runML():
    global X_train, X_test, y_train, y_test
    global machine_learning
    text.delete('1.0', END)
    cls = LinearRegression()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    #acc = accuracy_score(predict,y_test)
    text.insert(END,"Linear Regression Machine Learning Model Generated\n")
    machine_learning = cls
    
def predictPolicy():
    text.delete('1.0', END)
    global machine_learning

    dataset_file1 = filedialog.askopenfilename(initialdir="insurance_dataset")
    data1 = pd.read_csv(dataset_file1)
    data1['sex'] = data1['sex'].apply({'male':0,      'female':1}.get) 
    data1['smoker'] = data1['smoker'].apply({'yes':1, 'no':0}.get)
    data1['region'] = data1['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)
    data1 = data1.values
    predict = machine_learning.predict(data1)
    for i in range(len(data1)):
        text.insert(END,str(data1[i])+" Insurance Charges Predicted As : "+str(predict[i])+"\n\n")
    
def exitSystem():
    main.destroy()
    
def predictSentiments():
    text.delete('1.0', END)
    sentiment = SentimentIntensityAnalyzer()
    review = simpledialog.askstring("Enter your review on insurance policy and charges", "Please enter your reviews on insurance policy and charges", parent=main)

    sentiment_dict = sentiment.polarity_scores(review)
    opinion = ''
    if sentiment_dict['compound'] >= 0.05 :
        opinion = 'Positive'
    elif sentiment_dict['compound'] <= - 0.05 :
        opinion = 'Negative'
    else :
        opinion = 'Neutral'
    text.insert(END,"Your Review : "+review+"\n\n")
    text.insert(END,"Predicted Sentiment on your insurance review : "+opinion+"\n") 
    

font = ('times', 16, 'bold')
title = Label(main, text='AUTOMATING E-GOVERNMENT SERVICES WITH MACHINE LEARNING AND ARTIFICIAL INTELLIGENCE')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Insurance Dataset", command=loadData)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

exploreButton = Button(main, text="Explore Insurance Dataset", command=exploreData)
exploreButton.place(x=50,y=150)
exploreButton.config(font=font1) 

mlButton = Button(main, text="Run Machine Learning Algorithm", command=runML)
mlButton.place(x=380,y=150)
mlButton.config(font=font1) 

predictButton = Button(main, text="Predict BMI Based Insurance Charges", command=predictPolicy)
predictButton.place(x=680,y=150)
predictButton.config(font=font1)

sentimentButton = Button(main, text="Predict Sentiments On Insurance", command=predictSentiments)
sentimentButton.place(x=50,y=200)
sentimentButton.config(font=font1) 

exitButton = Button(main, text="Close Here", command=exitSystem)
exitButton.place(x=380,y=200)
exitButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='green')
main.mainloop()
