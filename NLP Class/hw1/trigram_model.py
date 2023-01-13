import sys
from collections import defaultdict, Counter
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    sequence1 = ["START"]*(n-1) + sequence + ["STOP"] if n > 2 else ["START"] + sequence + ["STOP"]
    index = 0
    end = 0 + n
    result = []
    while end < len(sequence1) + 1:
        result.append(tuple(sequence1[index:end]))
        index, end = index+1, end+1
    return result


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = Counter() # might want to use defaultdict or Counter instead
        self.bigramcounts = Counter() 
        self.trigramcounts = Counter()
        self.totalwords = 0#To be used for calculating raw_unigram_prob. Will include STOP, but not START tokens
        self.lines = 0
        for corpus_line in corpus:
            self.lines += 1
            self.totalwords += (len(corpus_line) + 1)#Total number of words is normal tokens + STOP tokens. Exclude START tokens
            self.unigramcounts.update(get_ngrams(corpus_line, 1))
            self.bigramcounts.update(get_ngrams(corpus_line, 2))
            self.trigramcounts.update(get_ngrams(corpus_line, 3))
        del self.unigramcounts[("START",)] #Delete the (START) unigram since START is not a part of the total words
            

        ##Your code here

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        #print(trigram, trigram[0:2])
        retval = 0.0
        if trigram[0:2] == ("START", "START"):
            return self.trigramcounts.get(trigram, 0) / self.lines
        try:
            retval = self.trigramcounts.get(trigram, 0) / self.bigramcounts.get(trigram[0:2], 0)
        except ZeroDivisionError:#If the bigram in the denominator hasn't been seen during training
            retval = self.raw_unigram_probability(trigram[2:])
        #print(retval)
        return retval

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        retval = 0.0
        if bigram[0:1] == ("START",):
            return self.bigramcounts.get(bigram, 0) / self.lines
        try:
            retval = self.bigramcounts.get(bigram, 0) / self.unigramcounts.get(bigram[0:1], 0)
        except ZeroDivisionError:#Used to prevent divide by Count(START) since we are omitting (START) unigram
            retval = self.raw_unigram_probability(bigram[1:])
        return retval
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        return self.unigramcounts.get(unigram, 0) / self.totalwords

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        '''
        lambda1*p(w|u,v) + lambda2*p(w|v) + lambda3*p(w)
        '''
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return (lambda1*self.raw_trigram_probability(trigram)) +\
            (lambda2*self.raw_bigram_probability(trigram[1:3])) +\
                (lambda3*self.raw_unigram_probability(trigram[2:]))
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        #Using trigrams
        trigrams = get_ngrams(sentence, 3)
            
        return sum(list(map(math.log2, list(map(self.smoothed_trigram_probability, trigrams)))))

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns perplexity of test corpus using training model.
        """
        #Iterate over all sentences and calculate total perplexity
        totalwords = 0
        sentence_logprobs = []
        for sentence in corpus:
            totalwords += len(sentence) + 1#Includes normal tokens and STOP tokens. Not START tokens 
            sentence_logprobs.append(self.sentence_logprob(sentence))
        return 2**(-1*(sum(sentence_logprobs) / totalwords))


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)#high
        model2 = TrigramModel(training_file2)#low

        total = len(os.listdir(testdir1)) + len(os.listdir(testdir2))
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp1 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp < pp1:
                correct += 1
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp < pp1:
                correct += 1
        
        return (correct/total)*100

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 
    
    assert get_ngrams(["natural","language","processing"],1) == \
[('START',), ('natural',), ('language',), ('processing',), ('STOP',)], "Value doesn't match"
    assert get_ngrams(["natural","language","processing"],2) == \
[('START', 'natural'), ('natural', 'language'), ('language', 'processing'), ('processing', 'STOP')], "Value doesn't match"
    assert get_ngrams(["natural","language","processing"],3) == \
[('START', 'START', 'natural'), ('START', 'natural', 'language'), ('natural', 'language', 'processing'), ('language', 'processing', 'STOP')],\
    "Value doesn't match"
    
    assert model.trigramcounts[('START','START','the')] == 5478, "Value doesn't match"
    
    assert model.bigramcounts[('START','the')] == 5478, "Value doesn't match"
    
    assert model.unigramcounts[('the',)] == 61428, "Value doesn't match"

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[1], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Training set perplexity:", pp)
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Testing set perplexity:", pp)


    # Essay scoring experiment: 
    #acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    acc = essay_scoring_experiment("hw1_data/ets_toefl_data/train_high.txt", 
                                   "hw1_data/ets_toefl_data/train_low.txt", 
                                   "hw1_data/ets_toefl_data/test_high", 
                                   "hw1_data/ets_toefl_data/test_low")
    
    print("Accuracy of Essay Classification:", acc)


