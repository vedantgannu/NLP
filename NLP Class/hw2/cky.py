"""
COMS W4705 - Natural Language Processing - Summer 2022
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""

import math
from re import L
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        #Create parse table and initialize for length 1 sentences
        parse_table = [[[]]*len(tokens) for _ in range(len(tokens))]
        for i in range(len(tokens)):
            parse_table[i][i] = self.grammar.rhs_to_rules[(tokens[i],)]
        
        for length in range(2, len(tokens)+1):#Iterate over span lengths
            for i in range(len(tokens) - length + 1):#Iterate over starting indices
                j = i + length
                for k in range(i+1, j):#Iterate over split points
                    left_cell_non_terminals = parse_table[i][k-1]
                    right_cell_non_terminals = parse_table[k][j-1]
                    if left_cell_non_terminals != [] and right_cell_non_terminals != []:#If spans from [i:k] and [k:j] possible
                        right_hand_rules = list(itertools.product(left_cell_non_terminals, right_cell_non_terminals))#Do cartesian product between both lists to get permutations of right hand rules
                        non_terminal_left_side = []
                        for right_hand_rule in right_hand_rules:
                            if (right_hand_rule[0][0], right_hand_rule[1][0]) in self.grammar.rhs_to_rules:
                                non_terminal_left_side += self.grammar.rhs_to_rules[(right_hand_rule[0][0], right_hand_rule[1][0])]
                        if len(parse_table[i][j-1]) == 0 and len(non_terminal_left_side) > 0:
                            parse_table[i][j-1] = non_terminal_left_side
                        else:
                            parse_table[i][j-1] += non_terminal_left_side
                    
        return True if len(parse_table[0][len(tokens)-1]) > 0 else False
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table= dict()
        probs = dict()
        
        #Create parse table and initialize for length 1 sentences
        for i in range(len(tokens)):
            # [('NP', ('NP', 'VP'), 0.00602409638554), ('S', ('NP', 'VP'), 0.694915254237), ('SBAR', ('NP', 'VP'), 0.166666666667), ('SQBAR', ('NP', 'VP'), 0.289156626506)]
            non_terminals = self.grammar.rhs_to_rules[(tokens[i],)]
            table[(i, i+1)] = dict()
            probs[(i,i+1)] = dict()
            for rule in non_terminals:
                table[(i, i+1)][rule[0]] = tokens[i]
                probs[(i,i+1)][rule[0]] = math.log2(rule[2])
        
        for length in range(2, len(tokens)+1):#Iterate over span lengths
            for i in range(len(tokens) - length + 1):#Iterate over starting indices
                j = i + length
                temp_table_i_j = []
                for k in range(i+1, j):#Iterate over split points
                    left_cell_non_terminals = table.get((i, k), None)
                    right_cell_non_terminals = table.get((k, j), None)
                    if left_cell_non_terminals != None and right_cell_non_terminals != None:#If spans from [i:k] and [k:j] possible
                        #Do cartesian product between keys to get permutations of right hand rules
                        right_hand_rules = list(itertools.product(left_cell_non_terminals.keys(), right_cell_non_terminals.keys()))
                        for right_hand_rule in right_hand_rules:
                            left_side_list = self.grammar.rhs_to_rules.get(right_hand_rule, [])
                            if len(left_side_list) != 0:#If the right produces non-terminal left side
                                for left_side in left_side_list:
                                    prob = math.log2(left_side[2])
                                    left_prob_cumulative = prob + probs[(i,k)][right_hand_rule[0]] + probs[(k,j)][right_hand_rule[1]]
                                    left_side_cumulative = (left_side[0], ( (right_hand_rule[0], i, k), (right_hand_rule[1], k, j) ), left_prob_cumulative)
                                    temp_table_i_j.append(left_side_cumulative)
                                    
                if len(temp_table_i_j) != 0:
                    t = defaultdict(list)
                    for items in temp_table_i_j:
                        t[items[0]].append(items)
                    for left_non_terminal in t:
                        max_left_rule = max(t[left_non_terminal], key=lambda rule: rule[2])
                        if table.get((i, j), None) == None:
                            table[(i, j)] = dict()
                        if probs.get((i, j), None) == None:
                            probs[(i, j)] = dict()
                        table[(i, j)][max_left_rule[0]] = max_left_rule[1]
                        probs[(i, j)][max_left_rule[0]] = max_left_rule[2]  
        
        if (0, len(tokens)) in table and (0, len(tokens)) in probs:
            return table, probs
        else:
            return None, None


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if not isinstance(chart[(i,j)][nt], tuple):#Base case
        return (nt, chart[(i,j)][nt])
    
    left_child = chart[(i,j)][nt][0]
    right_child = chart[(i,j)][nt][1]
    
    output = (nt, get_tree(chart, left_child[1], left_child[2], left_child[0]),
                    get_tree(chart, right_child[1], right_child[2], right_child[0]))
    return output 

 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        #toks =['miami', 'flights','cleveland', 'from', 'to','.']
        #toks = ['flights', 'from', 'los', 'angeles', 'to', 'pittsburgh', '.'] 
        print(toks) 
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        print()
        assert check_table_format(table) == True
        assert check_probs_format(probs) == True
        print(get_tree(table, 0, len(toks), grammar.startsymbol))
    
    
    '''
    with open('test.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['she', 'saw','the', 'cat', 'with','glasses']
        print(toks) 
        #print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
    '''
    
        
    
        
