"""
COMS W4705 - Natural Language Processing - Summer 2022 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum, isclose

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        #Check for correct format: Chomsky Normal Form, all left hand sides are capital an
        #Assume all non-terminals are uppercase and words are lowercase
        non_terminals = set()
        for lhs in self.lhs_to_rules:
            if not lhs.isupper():
                return False
            non_terminals.add(lhs)
        for lhs in self.lhs_to_rules:
            for rule_tuple in self.lhs_to_rules[lhs]:#Get all the associated rules for the left non-terminal symbol
                if len(rule_tuple[1]) > 1:
                    for symbol in rule_tuple[1]:#Check that each right hand symbol is a non-terminal
                        if symbol not in non_terminals:
                            return False
                    
        #Check that probabilities sum to 1 for each left side non-terminal symbol
        for lhs in self.lhs_to_rules:
            if not isclose(fsum([right_tuple[2] for right_tuple in self.lhs_to_rules[lhs]]), 1.0):
                return False
            
        return True 


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        print(grammar.lhs_to_rules)
        print()
        print()
        print(grammar.rhs_to_rules)
        if grammar.verify_grammar():
            print("PCFG is in CNF and valid")
        else:
            print("PCFG is either not in CNF or not valid")
        
        
        
        
