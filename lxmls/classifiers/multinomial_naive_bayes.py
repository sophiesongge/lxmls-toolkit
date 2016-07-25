import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
import sys
from lxmls.distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape
        print ("x.shape: ",x.shape)
        print ("y.shape: ",y.shape)

        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        # ----------
        # Solution to Exercise 1.1
        #count[i] is the number of documents in class i
        count = np.zeros(n_classes)
        #totalword[i] is the total number of words in class i
        totalword = np.zeros(n_classes)
        
        for i in range(n_classes):
            docs_in_class, _ = np.nonzero(y == classes[i])  # docs_in_class = indices of documents in class i
            #countword[k] is the number of word k in class i
            countword = np.zeros(n_words)
            for j in range(y.shape[0]):
                if(y[j] == classes[i]):
                    count[i] += 1
                    totalword[i] += n_words
                    for k in range(n_words):
                        countword[k] += x[j,k]
            prior[i] = count[i] / n_docs
            for k in range(n_words):
                likelihood[k, i] = countword[k] / totalword[i]

        params = np.zeros((n_words+1, n_classes))
        for i in range(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
