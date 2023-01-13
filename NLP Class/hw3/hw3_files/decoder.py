from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
from tensorflow import keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)

        while state.buffer: 
            #pass
            #Get representation of input for predicting transition state
            input_state = self.extractor.get_input_representation(words, pos, state).reshape((1, 6))
            softmax_output_vector = self.model.predict(input_state).flatten()
            #Grabbing only permitted transitions
            sorted_output = sorted([(index, prob)for index, prob in enumerate(softmax_output_vector.tolist())],
                                    key=lambda x: x[1], reverse=True)
            valid_index = None
            valid_action = None
            for element in sorted_output:
                if element[0] % 2 == 1:#ARC-LEFT
                    if len(state.buffer) > 0 and len(state.stack) > 0:
                        if state.stack[-1] != 0:
                            valid_index = element[0]
                            break
                elif element[0] % 2 == 0 and element[0] != 0:#ARC-RIGHT
                    if len(state.buffer) > 0 and len(state.stack) > 0:
                        valid_index = element[0]
                        break    
                else:#If a SHIFT operation
                    if len(state.buffer) == 1:
                        if len(state.stack) == 0:#Buffer shouldn't be empty after a SHIFT operation when stack has elements
                            valid_index = element[0]
                            break
                    elif len(state.buffer) > 1:#Can do a SHIFT as long as the buffer has elements
                        valid_index = element[0]
                        break
                        
            valid_action = self.output_labels[valid_index]
            if valid_action[0] == "shift":
                state.shift()
            elif valid_action[0] == "left_arc":
                state.left_arc(valid_action[1])
            elif valid_action[0] == "right_arc":
                state.right_arc(valid_action[1])
            
            
            

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])
    num = 0
    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
            # num += 1
            # print(num)
            
        
