import ast
import numpy as np 

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC


#abrindo arquivo de dados
def svm(vetDados):
    #arquivo de dados
    datas = []
    #arquivo de labels
    labels = []
    #arquivo de dicionario p/ svm
    ftrs = []

    with open(vetDados.data, 'r') as a:
        for data in a:
            vet = data.split(vetDados.separador)
            vet[len(vet)-1] = vet[len(vet)-1][:-1] # exclui o \n
            datas.append(vet[:vetDados.labelPosi] + vet[vetDados.labelPosi+1:])
            labels.append(vet[vetDados.labelPosi])

    tam = len(datas[0])

    for j in range(len(datas)):
        ftrs.append({i:datas[j][i] for i in range(tam)})

    #print(len(labels))

    vectorizer = DictVectorizer()
    features = vectorizer.fit_transform(ftrs)

    #Classificador
    clf = LinearSVC(dual=False, tol=1e-3)

    #validação cruzada
    pred = cross_val_predict(clf, features, labels, cv=10)

    return labels, pred



