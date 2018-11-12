from naiveBayes import *
from dataManip import *

class Teste:
    def __init__(self, data, nomeProb, labelPosi, div):
        self.data      = data 
        self.nomeProb  = nomeProb
        self.labelPosi = labelPosi
        self.separador = div

testes = [
    Teste("./dataset/cars/car.data", "car", 6, ','),
    Teste("./dataset/mushroom/agaricus-lepiota.data", "mushroom", 0, ","),
    Teste("./dataset/nursery/nursery.data", "nursery", 8, ',')
]

# Variaveis
numFolds = 10
tstAtl = 1

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


for i in range(numFolds):
    classifier.train("outputs/{0}_{1}_train.txt".format(testes[tstAtl].nomeProb, i))
    classifier.saveModel("{0}_{1}.model.txt".format(testes[tstAtl].nomeProb,i))
    classifier.test("outputs/{0}_{1}_test.txt".format(testes[tstAtl].nomeProb, i),  str(i) + "_" + testes[tstAtl].nomeProb)

#classifier.readFromModel("adult")
