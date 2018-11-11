import operator, os, shutil

class NaiveBayesClassifier:
    def __init__(self, separSymb, labelPosi):
        # frequencia de cada atributo de cada classe
        self.attFreq = {}
        # quantidade de amostras por classe
        self.classFreq = {}
        self.numSamples = 0
        # matriz de confusao
        self.confMatrix = []
        # simbolo de separacao
        self.separSymb = separSymb
        # posicao da label
        self.labelPosi = labelPosi
        

    # realiza o treino
    def train(self,  dataFile):
        print("Training...")

        dataFile = open(dataFile, 'r')
        # para todas as amostras no arquivo
        for sample in dataFile:
            features = sample.split(self.separSymb)
            tam = len(features)-1
            clazz = features[self.labelPosi]

            if self.labelPosi == tam:
                clazz = clazz[:-1]

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

        self.numSamples = sum(self.classFreq.values())

        print("Done! Samples read: {0}\n".format(self.numSamples))
        dataFile.close()

    def test(self,  dataFile, outFileName):
        if (not self.classFreq) or (not self.attFreq):
            print("Error: No training performed!")
        else:
            print("Testing...")
            count = 0
            dataFile = open(dataFile, 'r')
            outFile = open("outputs/"+outFileName+"_stats.test.txt", 'w')
            classes = sorted(list(self.classFreq.keys()))
            self.confMatrix = {i:{i:0 for i in classes} for i in classes}

            # prepara o arquivo de saida
            outFile.write("pred ")
            for classe in classes:
                outFile.write("{0} ".format(classe))
            outFile.write('\n')

            # para todas as amostras no arquivo
            for sample in dataFile:
                count += 1
                features = sample.split(self.separSymb)
                tam = len(features)-1
                clazz = features[self.labelPosi]

                if self.labelPosi == tam:
                    clazz = clazz[:-1]

                prob = {i:1.0 for i in classes}
                for classe in classes:
                    # realiza a produtoria
                    for i in range(tam):
                        # probabilidade de um atributo acontecer dentro de uma classe
                        if not features[i] in self.attFreq[classe][i].keys():
                            # se a chave nao existe, a probabilidade vai para zero
                            prob[classe] = 0.0
                            break
                        else:
                            prob[classe] *= self.attFreq[classe][i][features[i]] / self.classFreq[classe]
                    # probabilidade de ocorrencia de uma classe,
                    # multiplicada pela produtoria
                    prob[classe] *= self.classFreq[classe] / self.numSamples

                # pega a classe com a maior predicao
                predic = max(prob.items(), key=operator.itemgetter(1))[0]
                # adiciona na matriz de confusao
                self.confMatrix[clazz][predic] += 1
                # salva o arq de saida]
                outFile.write(predic + " ")
                for classe in classes:
                    if sum(prob.values()) == 0:
                        outFile.write("0.0000 ")
                    else:
                        outFile.write("{0} ".format(round(prob[classe]/sum(prob.values()), 4)))
                outFile.write('\n')
            
            print("Done!\nSamples tested: {0}\n".format(count))
            print(self.confMatrix)
            som = 0
            for classe in self.confMatrix.keys():
                som += self.confMatrix[classe][classe]
            print("Acuracia: " + str(som/count) )


    # salva os dois dicionarios em um arquivo
    def saveModel(self, fileName):
        print("Saving model to file...")
        outFile = open("outputs/"+fileName+".model.txt", 'w')
        outFile.write("{0}\n".format(self.classFreq))
        outFile.write("{0}\n".format(self.attFreq))
        outFile.close()
        print("Done! Model saved in 'outputs/"+fileName+".model.txt'")

    # le os dois dicionarios do arquivo
    def readFromModel(self, fileName):
        print("Reading Model from file: 'outputs/"+fileName+".model.txt'")
        modelFile = open("outputs/"+fileName+".model.txt", 'r')
        self.classFreq = eval(modelFile.readline())
        self.attFreq = eval(modelFile.readline())
        self.numSamples = sum(self.classFreq.values())
        modelFile.close()
        print('Done!')

    def cleanOutput(self):
        if os.path.isdir("./outputs/"):
            print("Diretorio de saida Limpado!")
            shutil.rmtree("./outputs/")
        
        os.mkdir("./outputs")
