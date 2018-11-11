from naiveBayes import *
from dataManip import *

numFolds = 2
nomeProb = "car"


# arquivo com os dados crus
data = "./dataset/cars/car.data"
# Classificador baseado em naive-bayes
classifier = NaiveBayesClassifier(',', 6)

# limpa as saidas
classifier.cleanOutput()

# classe que gera o arquivo de folds e processa os dados
# parametros = (numFolds, nomeArqSaida, arqEntrada)
dataMinipu = dataManip(numFolds, nomeProb, data)

# processa os dados
dataMinipu.formatData(0)
# Cria arquivo dos folds
dataMinipu.makeTestTrainFiles()


for i in range(numFolds):
    classifier.train("outputs/{0}_{1}_train.txt".format(nomeProb, i))
    classifier.saveModel(nomeProb + ".model.txt")
    classifier.test("outputs/{0}_{1}_test.txt".format(nomeProb, i),  str(i) + "_" + nomeProb)

#classifier.readFromModel("adult")
