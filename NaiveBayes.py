import sys
import getopt
import os
import math
from collections import defaultdict

class NaiveBayes:
    class TrainSplit:
        """
        Set of training and testing data
        """
        def __init__(self):
            self.train = []
            self.test = []

    class Document:
        """
        This class represents a document with a label. classifier is 'pos' or 'neg' while words is a list of strings.
        """
        def __init__(self):
            self.classifier = ''
            self.words = []

    def __init__(self):
        """
        Initialization of naive bayes
        """
        self.stopList = set(self.readFile('data/english.stop'))
        self.bestModel = False
        self.stopWordsFilter = False
        self.naiveBayesBool = False
        self.testModel = False
        self.numFolds = 10

        self.positiveReviews = 0
        self.negativeReviews = 0

        self.posWordCounts = defaultdict(lambda: 0)
        self.negWordCounts = defaultdict(lambda: 0)

        self.allCounts = defaultdict(lambda: 0)

        self.totalPosWords = 0
        self.totalNegWords = 0
        self.allWords = {""}


        # TODO
        # Implement a multinomial naive bayes classifier and a naive bayes classifier with boolean features. The flag
        # naiveBayesBool is used to signal to your methods that boolean naive bayes should be used instead of the usual
        # algorithm that is driven on feature counts. Remember the boolean naive bayes relies on the presence and
        # absence of features instead of feature counts.

        # When the best model flag is true, use your new features and or heuristics that are best performing on the
        # training and test set.

        # If any one of the flags filter stop words, boolean naive bayes and best model flags are high, the other two
        # should be off. If you want to include stop word removal or binarization in your best performing model, you
        # will need to write the code accordingly.

    def classify(self, words):
        """
        Classify a list of words and return a positive or negative sentiment
        """
        if self.stopWordsFilter:
            words = self.filterStopWords(words)

        #if self.bestModel:
            #words = self.filterStopWords(words)
            #words = self.negateWords(words)


        # P(C)
        totalReviews = (self.positiveReviews + self.negativeReviews)
        probNeg = math.log(float(self.negativeReviews) / float(totalReviews))
        probPos = math.log(float(self.positiveReviews) / float(totalReviews))

        conditionalNeg = 0
        conditionalPos = 0

        alpha = 1


        if self.naiveBayesBool:
            alpha = 1
            for word in list(set(words)):
                if self.allCounts[word] > 0:
                    # sum negative classifications
                    totalNegCounts = self.totalNegWords + alpha * len(self.allCounts)
                    conditionalNeg += math.log(float(self.negWordCounts[word] + alpha) / float(totalNegCounts))

                    # sum positive classifications
                    totalPosCounts = self.totalPosWords + alpha * len(self.allCounts)
                    conditionalPos += math.log(float(self.posWordCounts[word] + alpha) / float(totalPosCounts))

        elif self.bestModel:
            alpha = 5
            for word in list(set(words)):
                if self.allCounts[word] > 0:
                    # sum negative classifications
                    totalNegCounts = self.totalNegWords + alpha * len(self.allCounts)
                    conditionalNeg += math.log(float(self.negWordCounts[word] + alpha) / float(totalNegCounts))

                    # sum positive classifications
                    totalPosCounts = self.totalPosWords + alpha * len(self.allCounts)
                    conditionalPos += math.log(float(self.posWordCounts[word] + alpha) / float(totalPosCounts))

        else:
            for word in words:
                if self.allCounts[word] > 0:
                    # sum negative classifications
                    totalNegCounts = self.totalNegWords + alpha * len(self.allCounts)
                    conditionalNeg += math.log(float(self.negWordCounts[word] + alpha) / float(totalNegCounts))

                    # sum positive classifications
                    totalPosCounts = self.totalPosWords + alpha * len(self.allCounts)
                    conditionalPos += math.log(float(self.posWordCounts[word] + alpha) / float(totalPosCounts))


        # P(C) * P(f | C)
        negScore = (probNeg + conditionalNeg)
        posScore = (probPos + conditionalPos)

        # classify a list of words and return the 'pos' or 'neg' classification
        if posScore > negScore:
            return 'pos'
        return 'neg'

    def addDocument(self, classifier, words):
        """
        Train model on a document with label classifier (pos or neg) and words (list of strings).
        """

        if self.naiveBayesBool:
            if classifier == 'pos':
                self.positiveReviews += 1
                for word in list(set(words)): # don't look at duplicate words
                    self.posWordCounts[word] += 1
                    self.totalPosWords += 1
                    self.allCounts[word] += 1

            elif classifier == 'neg':
                self.negativeReviews += 1
                for word in list(set(words)):
                    self.negWordCounts[word] += 1
                    self.totalNegWords += 1
                    self.allCounts[word] += 1

        elif self.bestModel:
            if classifier == 'pos':
                self.positiveReviews += 1 # same as "positive documents"

                for word in list(set(words)): # don't look at duplicate words
                    self.posWordCounts[word] += 1
                    self.totalPosWords += 1
                    self.allCounts[word] += 1
                    self.allWords.add(word)

            elif classifier == 'neg':
                self.negativeReviews += 1
                for word in list(set(words)):
                    self.negWordCounts[word] += 1
                    self.totalNegWords += 1
                    self.allCounts[word] += 1
                    self.allWords.add(word)

        else:
            if classifier == 'pos':
                self.positiveReviews += 1
                for word in words:
                    self.posWordCounts[word] += 1
                    self.totalPosWords += 1
                    self.allCounts[word]+=1

            elif classifier == 'neg':
                self.negativeReviews += 1
                for word in words:
                    self.negWordCounts[word] += 1
                    self.totalNegWords += 1
                    self.allCounts[word]+=1

        pass

    def readFile(self, fileName):
        """
        Reads a file and segments.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        str = '\n'.join(contents)
        result = str.split()
        return result

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        for fileName in posDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            doc.classifier = 'pos'
            split.train.append(doc)
        for fileName in negDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            doc.classifier = 'neg'
            split.train.append(doc)
        return split

    def train(self, split):
        for doc in split.train:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)

            self.addDocument(doc.classifier, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            yield split

    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for doc in split.test:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)

            guess = self.classify(words)
            labels.append(guess)
        return labels

    def buildSplits(self, args):
        """
        Construct the training/test split
        """
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print '[INFO]\tOn %d-fold of CV with \t%s' % (self.numFolds, trainDir)

            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    doc.classifier = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                for fileName in negDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    doc.classifier = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                split.train.append(doc)

            posDocTest = os.listdir('%s/pos/' % testDir)
            negDocTest = os.listdir('%s/neg/' % testDir)
            for fileName in posDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                doc.classifier = 'pos'
                split.test.append(doc)
            for fileName in negDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                doc.classifier = 'neg'
                split.test.append(doc)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """
        Stop word filter
        """
        removed = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                removed.append(word)
        return removed

    def negateWords(self, words):
        """
        Deals with negation
        """
        newWords = []
        negate = False
        for word in words:
            # if word contains n't, not, no, never
            if "n't" in word or word.startswith("non-") \
                    or word == "not" or word == "no" or word =="never" or word == "neither" or word == "nor" or word == "but":
                negate = True
            if negate:
                if word not in (',', '.', '?', '!', ';'):
                    word = "NOT_" + word
                else:
                    negate = False
            newWords.append(word)
        return newWords


def test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = NaiveBayes()
        classifier.stopWordsFilter = stopWordsFilter
        classifier.naiveBayesBool = naiveBayesBool
        classifier.bestModel = bestModel
        accuracy = 0.0
        for doc in split.train:
            words = doc.words
            classifier.addDocument(doc.classifier, words)

        for doc in split.test:
            words = doc.words
            guess = classifier.classify(words)
            if doc.classifier == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyFile(stopWordsFilter, naiveBayesBool, bestModel, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.stopWordsFilter = stopWordsFilter
    classifier.naiveBayesBool = naiveBayesBool
    classifier.bestModel = bestModel
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print classifier.classify(testFile)


def main():
    stopWordsFilter = False
    naiveBayesBool = False
    bestModel = False
    testModel = False #fixme delete later


    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f', '') in options:
        stopWordsFilter = True
    elif ('-b', '') in options:
        naiveBayesBool = True
    elif ('-m', '') in options:
        bestModel = True
    # elif ('-m', '') in options:
    #     #stopWordsFilter = True
    #     testModel = True

    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(stopWordsFilter, naiveBayesBool, bestModel, args[0], args[1])
    else:
        test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel)


if __name__ == "__main__":
    main()


 # Bernoulli model
            # testWords = list(set(words)) # create list of words in test set
            # for trainingWord in self.allWords: # calculate conditionalNeg and conditionalPos
            #     posNum = float(self.posWordCounts[trainingWord]) + 1
            #     negNum = float(self.negWordCounts[trainingWord]) + 1
            #
            #     posDenom = float(self.positiveReviews) + 2
            #     negDenom = float(self.negativeReviews) + 2
            #
            #     posProb = posNum/posDenom
            #     negProb = negNum/negDenom
            #
            #     if trainingWord in testWords: # multiply its probability of occurrence
            #         conditionalPos += math.log(posProb)
            #         conditionalNeg += math.log(negProb)
            #
            #     else: # multiply 1 - probabilities of word
            #         conditionalPos += math.log(1-posProb)
            #         conditionalNeg += math.log(1-negProb)