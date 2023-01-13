#!/usr/bin/env python
from collections import defaultdict
import sys
import itertools

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers

from typing import List
import string

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    ret = set()
    for synset in wn.synsets(lemma, pos=pos):
        for lexeme in synset.lemmas():
            if lexeme.name() == lemma:
                continue
            elif lexeme.name().find("_") != -1:
                ret.add(lexeme.name().replace("_", " "))
            else:
                ret.add(lexeme.name())
    return list(ret)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    #return None # replace for part 2
    #Context obj: self.cid, self.word_form, self.lemma, self.pos, self.left_context, self.right_context
    synsets = wn.synsets(context.lemma, pos=context.pos)#Get the synonym set that the input word relates to
    frequency = defaultdict(int)
    for synset in synsets:
        for lexeme in synset.lemmas():
            if lexeme.name() != context.lemma:#Consider the lemmas that aren't the input lemma (synonyms)
                frequency[lexeme.name()] += lexeme.count()#Record the number of occurences for each synonym sharing word sense with input lemma
                
    return max(frequency, key=frequency.get).replace("_", " ")
        
        
        

def wn_simple_lesk_predictor(context : Context) -> str:
    synsets = wn.synsets(context.lemma, pos=context.pos)#Get the synonym set that the input word relates to
    stop_words = stopwords.words('english')
    max_overlap = 0
    overlap_dict = defaultdict(int)
    for synset in synsets:
        #Tokenize and filter out stop words of synset definition, left context, right context, and examples
        definitions = [ [word.lower() for word in tokenize(synset.definition()) if word.lower() not in stop_words] ]
        left_context = tokenize(" ".join([word.lower() for word in context.left_context if word.lower() not in stop_words]))
        right_context = tokenize(" ".join([word.lower() for word in context.right_context if word.lower() not in stop_words]))
        examples = []
        for example in synset.examples():
            examples.append([word.lower() for word in tokenize(example) if word.lower() not in stop_words])
        #Do same filtering and tokenization for hypernym definitions and examples
        for synset_hyper in synset.hypernyms():
            definitions.append( [word.lower() for word in tokenize(synset_hyper.definition()) if word.lower() not in stop_words] )
            for example in synset_hyper.examples():
                examples.append([word.lower() for word in tokenize(example) if word.lower() not in stop_words])
        #Get the overlap
        overlap = 0
        for gloss in definitions + examples:
            overlap += len(set(gloss) & set(left_context)) + len(set(gloss) & set(right_context))
        overlap_dict[synset] = overlap
        if overlap > max_overlap:
            max_overlap = overlap
    best_synsets = [synset for (synset, overlap) in overlap_dict.items() if overlap == max_overlap]
    lexemes = []
    best_synset_frequency = 0
    synset_dict = dict()
    for synset in (best_synsets if best_synsets else synsets):#Should handle if overlap/no overlap exists
        lexemes = []
        lex_count = 0
        for lexeme in synset.lemmas():
            if lexeme.name() == context.lemma:
                lex_count = lexeme.count()
                if lexeme.count() > best_synset_frequency:
                    best_synset_frequency = lexeme.count()
            else:
                lexemes.append(lexeme)
        if lexemes:
            synset_dict[synset] = (lex_count, lexemes)
        
    most_frequent_synsets = [(synset, lex[1]) for synset, lex in synset_dict.items() if lex[0] == best_synset_frequency]
    #Select most frequent lexeme from synset(s)
    if most_frequent_synsets:
        return max(itertools.chain(*[lexemes for _, lexemes in most_frequent_synsets]), key=lambda lexeme: lexeme.count()).name().replace("_", " ")
    else:
        return "smurf"
                
            
    #return None #replace for part 3
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        syns = get_candidates(context.lemma, context.pos)
        lemmas = []
        for syn in syns:
            try:
                lemmas.append((syn, self.model.similarity(context.lemma, syn.replace(" ",  "_"))))
            except:
                continue
        if lemmas:
             return max(lemmas, key=lambda lemma: lemma[1])[0]
        else:
            return "smurf"
        #return None # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        #self.wv_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

    def predict(self, context : Context) -> str:
        syns = get_candidates(context.lemma, context.pos)
        left_context = ''
        for word in context.left_context:
            if word.isalpha():
                left_context = left_context + ' ' + word
            else:#Handle things like "do n't -> don't"    ;    Dubhghall , from -> Dubhghall, from
                left_context = left_context + word

        sentence = left_context + ' [MASK]'

        right_context = ''
        for word in context.right_context:
            if word.isalpha():
                right_context = right_context + ' ' + word
            else:#Handle things like "do n't -> don't"    ;    Dubhghall , from -> Dubhghall, from
                right_context = right_context + word
        
        sentence = sentence + right_context

        input_toks_encoded = self.tokenizer.encode(sentence)
        mask_index = self.tokenizer.convert_ids_to_tokens(input_toks_encoded).index('[MASK]')
        input_mat = np.array(input_toks_encoded).reshape((1,-1))
        outputs = self.model.predict(input_mat, verbose=0)
        predictions = outputs[0]
        best_words_indices = np.argsort(predictions[0][mask_index])[::-1] # Sort in increasing order
        best_words = self.tokenizer.convert_ids_to_tokens(best_words_indices)
        for word in best_words:
            if word.replace("_", " ") in syns:
                return word.replace("_", " ")
        return ""
        #return None # replace for part 5
    
    def predict2(self, context : Context) -> str:
        syns = get_candidates(context.lemma, context.pos)
        left_context = ''
        for word in context.left_context:
            if word.isalpha():
                left_context = left_context + ' ' + word
            else:#Handle things like "do n't -> don't"    ;    Dubhghall , from -> Dubhghall, from
                left_context = left_context + word

        sentence = left_context + ' [MASK]'

        right_context = ''
        for word in context.right_context:
            if word.isalpha():
                right_context = right_context + ' ' + word
            else:#Handle things like "do n't -> don't"    ;    Dubhghall , from -> Dubhghall, from
                right_context = right_context + word
        
        sentence = sentence + right_context
        
        sentence_unmask = left_context + right_context

        input_toks_encoded = self.tokenizer.encode(sentence_unmask) + self.tokenizer.encode(sentence)[1:]
        mask_index = self.tokenizer.convert_ids_to_tokens(input_toks_encoded).index('[MASK]')
        input_mat = np.array(input_toks_encoded).reshape((1,-1))
        outputs = self.model.predict(input_mat, verbose=0)
        predictions = outputs[0]
        best_words_indices = np.argsort(predictions[0][mask_index])[::-1] # Sort in increasing order
        best_words = self.tokenizer.convert_ids_to_tokens(best_words_indices)
        for word in best_words:
            if word.replace("_", " ") in syns:
                return word.replace("_", " ")
        return ""
    
        # best_word = ""
        # scores = []
        # i = 0
        # for word in best_words:
        #     if i > 10:
        #         break
        #     #print(i, word)
        #     word_ = best_words[i]
        #     #word_ = word.replace("_", " ")
        #     if word_.replace("_", " ") != context.lemma:
        #         score = 0
        #         for syn in syns:
        #             try:
        #                 score += wv_model.similarity(word_, syn.replace(" ",  "_"))
        #             except:
        #                 continue
        #         scores.append((word_, score/len(syns)))
        #         i += 1
        #     else:
        #         continue
        
        # if scores:
        #     return max(scores)[0].replace("_", " ")
        # else:
        #     return "smurf"


    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    
    # for context in read_lexsub_xml(sys.argv[1]):
    #     #print(context)  # useful for debugging
    #     prediction = smurf_predictor(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    
    # for context in read_lexsub_xml(sys.argv[1]):
    #     #print(context)  # useful for debugging
    #     prediction = wn_frequency_predictor(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    
    # for context in read_lexsub_xml(sys.argv[1]):
    #     #print(context)  # useful for debugging
    #     prediction = wn_simple_lesk_predictor(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    
    # W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)
    # for context in read_lexsub_xml(sys.argv[1]):
    #     #print(context)  # useful for debugging
    #     prediction = predictor.predict_nearest(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    
    # predictor = BertPredictor()
    # for context in read_lexsub_xml("lexsub_trial.xml"):
    #     #print(context)  # useful for debugging
    #     prediction = predictor.predict(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    
    # for context in read_lexsub_xml(sys.argv[1]):
    #     #print(context)  # useful for debugging
    #     prediction = part3(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    
    #Combines unmasked sentence with masked sentence as input to BERT, in
    #an attempt to see if the knowledge about the original word has increased bearing
    #on the best available substitution word 
    # as demonstrated in paper:
    #https://arxiv.org/pdf/1907.06226v1.pdf
    #I tried to implement a scoring algo with word vectors, but couldn't get it to work. It was commented out
    predictor = BertPredictor()
    for context in read_lexsub_xml("lexsub_trial.xml"):
        #print(context)  # useful for debugging
        prediction = predictor.predict2(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
