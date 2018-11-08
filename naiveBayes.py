class NaiveBayesClassifier:
    def __init__(self):
        # frequencia de cada atributo de cada classe
        self.attFreq = {}
        # quantidade de amostras por classe
        self.classFreq = {}
        # matriz de confusao
        self.confMatrix = []

    # realiza o treino
    def train(self,  dataFile, separSymb):
        print("Training...")

        dataFile = open(dataFile, 'r')
        # para todas as amostras no arquivo
        for sample in dataFile:
            features = sample.split(separSymb)
            clazz = features[len(features)-1][:-1]
            tam = len(features)-1

            # caso a classe nao exista nos dicionarios
            if clazz not in self.attFreq.keys():
                self.attFreq[clazz] = [{} for i in range(tam)]

            if clazz not in self.classFreq.keys():
                self.classFreq[clazz] = 0

            # conta a frequencia da classe
            self.classFreq[clazz] += 1

            # para todos os features da amostra
            for i in range(tam):
                # se o atributo nao existir no dicionario
                if features[i] not in self.attFreq[clazz][i].keys():
                    self.attFreq[clazz][i][features[i]] = 0

                # conta a frequencia do atributo
                self.attFreq[clazz][i][features[i]] += 1

        print("Done! Samples read: {0}".format(sum(self.classFreq.values())))
        dataFile.close()

    # salva os dois dicionarios em um arquivo
    def saveModel(self):
        print("Saving model to file...")
        outFile = open("outputs/cards.model.txt", 'w')
        outFile.write("{0}\n".format(self.classFreq))
        outFile.write("{0}\n".format(self.attFreq))
        outFile.close()
        print("Done! Model saved in 'outputs/cards.model.txt'")

    # le os dois dicionarios do arquivo
    def readFromModel(self):
        print("Reading Model from file: 'outputs/cards.model.txt'")
        modelFile = open("outputs/cards.model.txt", 'r')
        self.classFreq = eval(modelFile.readline())
        self.attFreq = eval(modelFile.readline())
        modelFile.close()
        print('Done!')