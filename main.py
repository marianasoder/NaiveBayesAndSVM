from naiveBayes import *

classifier = NaiveBayesClassifier()
trainFile = "./dataset/poker-hand-training-true.data"

classifier.train(trainFile, ',')
classifier.saveModel()
classifier.readFromModel()