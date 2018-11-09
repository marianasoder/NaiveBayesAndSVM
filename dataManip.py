class dataManip:
    def __init__(self, nFlds, otFil, dtFil):
        self.numFolds = nFlds
        self.outName = otFil
        self.outFile = "outputs/{0}_{1}folds.txt".format(otFil, nFlds)
        self.dataFile = dtFil

    def formatData(self):
        arq = open(self.dataFile, 'r')
        featList = {}

        print("Reading data...")
        for line in arq:
            features = line.split(',')
            tam = len(features)-1
            clas = features[tam][:-1]

            if clas not in featList.keys():
                featList[clas] = []

            ret = ""
            for i in range(1, tam):
                ret += "{0},".format(features[i])
            ret += features[tam]

            featList[clas].append(ret)
        arq.close()

        print("Creating folds")
        saida = open(self.outFile, 'w')
        for clas in featList.keys():
            i = 0
            for fold in self.__kFoldsGen(featList[clas]):
                for element in fold:
                    saida.write("{0} {1}".format(i, element))
                i += 1
        saida.close()
        print("Done!")

    def __kFoldsGen(self, vector):
        return [vector[i::self.numFolds] for i in range(self.numFolds)]

    def makeTestTrainFiles(self):
        arq = open(self.outFile, 'r')
        filesTest = [open("outputs/{0}_{1}_test".format(self.outName, i), 'w') for i in range(self.numFolds)]
        filesTrain = [open("outputs/{0}_{1}_train".format(self.outName, i), 'w') for i in range(self.numFolds)]

        print("Criando arquivos de folds...")
        for line in arq:
            fold, features = line.split(" ")

            filesTest[int(fold)].write(features)

            for i in [j for j in range(self.numFolds) if j != int(fold)]:
                filesTrain[i].write(features)
        
        arq.close()
        for i in range(self.numFolds):
            filesTest[i].close()
            filesTrain[i].close()

        print("Done!")