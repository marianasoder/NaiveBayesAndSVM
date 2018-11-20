class dataManip:
    def __init__(self, nFlds, otFil, dtFil, labelPosi, divisor):
        self.numFolds = nFlds
        self.outName = otFil
        self.outFile = "outputs/{0}_{1}_folds.txt".format(otFil, nFlds)
        self.dataFile = dtFil
        self.labelPosi = labelPosi
        self.divSym = divisor

    def formatData(self):
        arq = open(self.dataFile, 'r')
        featList = {}

        # Separa as amostras por classe
        #print("Reading data...")
        for line in arq:
            features = line.split(self.divSym)
            tam = len(features)-1
            clas = features[self.labelPosi]

            if self.labelPosi == tam:
                clas = clas[:-1]

            if clas not in featList.keys():
                featList[clas] = []

            featList[clas].append(line)
        arq.close()

        # Divide em folds
        #print("Creating folds")
        saida = open(self.outFile, 'w')
        for clas in featList.keys():
            i = 0
            # eliminar as repeticoes
            #featList[clas] = list(set(featList[clas]))
            for fold in self.__kFoldsGen(featList[clas]):
                for element in fold:
                    saida.write("{0} {1}".format(i, element))
                i += 1
        saida.close()
        #print("Done!\n")

    def __kFoldsGen(self, vector):
        return [vector[i::self.numFolds] for i in range(self.numFolds)]

    # cria arquivos de teste e treino
    def makeTestTrainFiles(self):
        arq = open(self.outFile, 'r')
        filesTest = [open("outputs/{0}_{1}_test.txt".format(self.outName, i), 'w') for i in range(self.numFolds)]
        filesTrain = [open("outputs/{0}_{1}_train.txt".format(self.outName, i), 'w') for i in range(self.numFolds)]

        #print("Criando arquivos de folds...")
        for line in arq:
            fold, features = line.split(" ")

            filesTest[int(fold)].write(features)

            for i in [j for j in range(self.numFolds) if j != int(fold)]:
                filesTrain[i].write(features)
        
        arq.close()
        for i in range(self.numFolds):
            filesTest[i].close()
            filesTrain[i].close()

        #print("Done!\n")