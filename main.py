from naiveBayes import *
from dataManip import *
from svm import *
import sys

from sklearn.metrics import confusion_matrix, classification_report

def printAnalysis(labels, preds, name):
    print("\n=================== RESULTADO "+name+" ===================")
    print(classification_report(labels, preds, digits=4))
    confMat = confusion_matrix(labels, preds)
    tam = len(confMat)
    acertos = sum([confMat[i][i] for i in range(tam)])
    acuracia = acertos / sum([confMat[i][j] for i in range(tam) for j in range(tam)])
    print("Accuracy = {0}\n".format(acuracia))
    print(confMat)

def Analysis(matConfs):
    labels = []
    preds = []

    for matConf in matConfs:
        for label in matConf.keys():
            for pred in matConf[label].keys():
                count = matConf[label][pred]
                labels.extend([label for i in range(count)])
                preds.extend([pred for i in range(count)])

    printAnalysis(labels, preds, "NAIVE-BAYES")    

class Teste:
    def __init__(self, data, nomeProb, labelPosi, div):
        self.data      = data 
        self.nomeProb  = nomeProb
        self.labelPosi = labelPosi
        self.separador = div

testes = [
    Teste("./dataset/cars/car.data", "car", 6, ','),
    Teste("./dataset/mushroom/agaricus-lepiota.data", "mushroom", 0, ","),
    Teste("./dataset/nursery/nursery2.data", "nursery", 8, ',')
]

# Variaveis
numFolds = 10

if len(sys.argv) < 2:
    tstAtl = 2
else:
    tstAtl = int(sys.argv[1])

# Classificador baseado em svm
predSvm, labelsSvm = svm(testes[tstAtl])
printAnalysis(predSvm, labelsSvm, "SVM")  

# Classificador baseado em naive-bayes
classifier = NaiveBayesClassifier(testes[tstAtl].separador, 
                                  testes[tstAtl].labelPosi)


# limpa as saidas
classifier.cleanOutput()

# classe que gera o arquivo de folds e processa os dados
# parametros = (numFolds, nomeArqSaida, arqEntrada)
dataMinipu = dataManip(numFolds, 
                       testes[tstAtl].nomeProb, 
                       testes[tstAtl].data, 
                       testes[tstAtl].labelPosi, 
                       testes[tstAtl].separador)

# processa os dados
dataMinipu.formatData()
# Cria arquivo dos folds
dataMinipu.makeTestTrainFiles()

matConfs = []
for i in range(numFolds):
    classifier.train("outputs/{0}_{1}_train.txt".format(testes[tstAtl].nomeProb, i))
    classifier.saveModel("{0}_{1}.model.txt".format(testes[tstAtl].nomeProb,i))
    matConfs.append(
        classifier.test("outputs/{0}_{1}_test.txt".format(testes[tstAtl].nomeProb, i),  str(i) + "_" + testes[tstAtl].nomeProb)
    )

Analysis(matConfs)
#classifier.readFromModel("adult")
