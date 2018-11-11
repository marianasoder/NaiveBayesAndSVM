from naiveBayes import *
from dataManip import *

numFolds = 4
nomeProb = "breast_cancer"

# arquivo com os dados crus
data = "./dataset/breast-cancer-wisconsin.data"
# Classificador baseado em naive-bayes
classifier = NaiveBayesClassifier()
# classe que gera o arquivo de folds e processa os dados
# parametros = (numFolds, nomeArqSaida, arqEntrada)
dataMinipu = dataManip(numFolds, nomeProb, data)

# processa os dados
dataMinipu.formatData(0)
# Cria arquivo dos folds
dataMinipu.makeTestTrainFiles()


for i in range(numFolds):
    classifier.train("outputs/{0}_{1}_train.txt".format(nomeProb, i), ',', 0)
    classifier.test("outputs/{0}_{1}_test.txt".format(nomeProb, i), ',', str(i) + "_" + nomeProb, 0)

#classifier.saveModel("adult")
#classifier.readFromModel("adult")
