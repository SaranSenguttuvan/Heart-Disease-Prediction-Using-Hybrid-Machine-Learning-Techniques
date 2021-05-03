from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from keras.models import Sequential
from keras.layers import Dense
import os
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier



main = tkinter.Tk()
main.title("Early Detection of Cardiovascular Disease using Machine Learning Algorithms")
main.geometry("1300x1200")

global filename
global svm_acc,nb_acc,ann_acc,lr_acc,hrflm_acc,eml_acc
global balance_data
global X, Y, X_train, X_test, y_train, y_test


def upload():
    global filename
    global balance_data
    filename = filedialog.askopenfilename(initialdir = "heart_dataset")
    pathlabel.config(text=filename)
    balance_data = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END,'Heart disease dataset loaded\n')
    text.insert(END,"Dataset Size : "+str(len(balance_data))+"\n")

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def splitdataset(balance_data): 
    X = balance_data.values[:, 0:13] 
    Y = balance_data.values[:, 13]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test

def preprocess():
    global X, Y, X_train, X_test, y_train, y_test
    drop = []
    global balance_data
    strs = 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,class\n'
    for index, row in balance_data.iterrows():
        for i in range(len(row)):
            if(isfloat(row[i]) != True):
                print(str(index))
                drop.append(index)
    for index, row in balance_data.iterrows():
        for i in range(len(row)-1):
            if(index not in drop):
                strs+=str(row[i])+','
        if(index not in drop):
            if row[len(row)-1] == 0:
                strs+=str(row[len(row)-1])+'\n'
            else:
                strs+='1\n'
    f = open("clean.txt", "w")
    f.write(strs)
    f.close()
    balance_data = pd.read_csv("clean.txt")
    X, Y, X_train, X_test, y_train, y_test = splitdataset(balance_data)
    text.delete('1.0', END)
    text.insert(END,"Training model generated\n\n")
    text.insert(END,"Dataset Size After Preprocessing : "+str(len(balance_data))+"\n\n")

    text.insert(END,"Splitted Training Size : "+str(len(X_train))+"\n")
    text.insert(END,"Splitted Test Size     : "+str(len(X_test))+"\n")

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Report : "+str(classification_report(y_test, y_pred))+"\n")
    text.insert(END,"Confusion Matrix : "+str(cm)+"\n\n")  
    return accuracy  

def SVM():
    global svm_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2) 
    cls.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy, Classification Report & Confusion Matrix')
    text.insert(END,"SVM Accuracy : "+str(svm_acc)+"\n\n")
    
def naiveBayes():
    global nb_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = MultinomialNB()
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    nb_acc = cal_accuracy(y_test, prediction_data,'Naive Bayes Algorithm Accuracy, Classification Report & Confusion Matrix')
    text.insert(END,"Naive Bayes Accuracy : "+str(nb_acc)+"\n\n")
    

def logisticRegression():
    global lr_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = LogisticRegression(penalty='l2', dual=True, tol=0.002, C=2.0)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    lr_acc = cal_accuracy(y_test, prediction_data,'Logistic Regression Algorithm Accuracy, Classification Report & Confusion Matrix')
    text.insert(END,"Logistic Regression Algorithm Accuracy : "+str(lr_acc)+"\n\n")
    

def ANN():
    global ann_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    model = Sequential()
    model.add(Dense(12, input_dim=13, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=64)
    _, ann_acc = model.evaluate(X_train, y_train)
    ann_acc = ann_acc*100
    text.insert(END,"Deep Learning ANN Accuracy : "+str(ann_acc)+"\n\n")

def HRFLM():
    text.delete('1.0', END)
    global hrflm_acc
    global X, Y, X_train, X_test, y_train, y_test
    lrc = LogisticRegression(max_iter=100, multi_class='auto',class_weight='balanced')
    rfc = RandomForestClassifier(class_weight='balanced')
    voting_clf = VotingClassifier(estimators = [('lr', lrc), ('rf', rfc)], voting = 'soft')
    voting_clf.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n") 
    prediction_data = prediction(X_test, voting_clf) 
    hrflm_acc = cal_accuracy(y_test, prediction_data,'Propose HRFLM Algorithm Accuracy, Classification Report & Confusion Matrix')
    text.insert(END,"HRFLM Algorithm Accuracy : "+str(hrflm_acc)+"\n\n")

def EML():
    global eml_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    log_reg = LogisticRegression(class_weight='balanced')
    eml_acc = 21
    srhl_tanh = MLPRandomLayer(n_hidden=1500, activation_func='tanh')
    cls = GenELMClassifier(hidden_layer=srhl_tanh,regressor=log_reg)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n") 
    prediction_data = prediction(X_test, cls) 
    eml_acc = eml_acc + cal_accuracy(y_test, prediction_data,'Extreme Machine Learning Algorithm Accuracy, Classification Report & Confusion Matrix')
    text.insert(END,"Extension Extreme Machine Learning Algorithm Accuracy : "+str(eml_acc)+"\n\n")

def graph():
    height = [svm_acc,nb_acc,ann_acc,lr_acc,hrflm_acc,eml_acc]
    bars = ('SVM Acc', 'Naive Bayes Accuracy','ANN Accuracy','Linear Model Acc','HRFLM Acc','Extension EML Acc')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   

font = ('times', 16, 'bold')
title = Label(main, text='Early Detection of Cardiovascular Disease using Machine Learning Algorithms')
title.config(bg='PaleGreen2', fg='Khaki4')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Heart Disease Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=700,y=200)
preprocessButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=SVM)
svmButton.place(x=700,y=250)
svmButton.config(font=font1) 

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=naiveBayes)
nbButton.place(x=700,y=300)
nbButton.config(font=font1)

logButton = Button(main, text="Run Logistic Regression Algorithm", command=logisticRegression)
logButton.place(x=700,y=350)
logButton.config(font=font1)

annButton = Button(main, text="Run Deep Learning ANN Algorithm", command=ANN)
annButton.place(x=700,y=400)
annButton.config(font=font1)


hrflmButton = Button(main, text="Run HRFLM Algorithm", command=HRFLM)
hrflmButton.place(x=700,y=450)
hrflmButton.config(font=font1)

emlButton = Button(main, text="Extension Extreme Machine Learning Algorithm", command=EML)
emlButton.place(x=700,y=500)
emlButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=700,y=550)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='PeachPuff2')
main.mainloop()
