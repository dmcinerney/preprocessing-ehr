import pandas as pd
import pickle as pkl
import networkx as nx

"""
This script processes the codes by arranging them into two data structures:
1. A dictionary mapping from code to node name in the graph
2. A networkx graph
"""

icd9_file = '/home/jered/Documents/data/icd_codes/diagnoses/icd9.txt'
icd10_file = '/home/jered/Documents/data/icd_codes/diagnoses/icd10.txt'
icd9to10_file = '/home/jered/Documents/data/icd_codes/diagnoses/icd9to10.csv'
icd10to9_file = '/home/jered/Documents/data/icd_codes/diagnoses/icd10to9.csv'
code_mapping_file = '/home/jered/Documents/data/icd_codes/code_mapping.pkl'
code_graph_file = '/home/jered/Documents/data/icd_codes/code_graph.pkl'

def process_file(filename1, filename2, split_at, encoding='utf-8'):
    with open(filename1, 'r', encoding=encoding) as file:
        with open(filename2, 'w') as file2:
            for i,line in enumerate(file):
                file2.write(line[:split_at].strip()+'\t'+line[split_at:].strip()+'\n')

def main():
    process_file(icd9_file, icd9_file+'_processed.csv', 6, encoding="ISO-8859-1")
    icd9 = pd.read_csv(icd9_file+'_processed.csv', names=['code', 'description'], delimiter='\t')
    process_file(icd10_file, icd10_file+'_processed.csv', 8)
    icd10 = pd.read_csv(icd10_file+'_processed.csv', names=['code', 'description'], delimiter='\t')
    icd9to10 = pd.read_csv(icd9to10_file)
    #icd10to9 = pd.read_csv(icd10to9_file)
    G = nx.DiGraph()
    # make icd10 graph and start code mapping
    code_mapping = {}
    non_leaf_codes = []
    for l in range(7, 2, -1):
        if l >= 4:
            for code in non_leaf_codes:
                G.add_edge(str(('ICD10',code[:l-1])),str(('ICD10',code)))
        non_leaf_codes = []
        for i,row in icd10[icd10.code.str.len() == l].iterrows():
            curr_node = str(('ICD10',row.code))
            G.add_node(curr_node)
            if l >= 4:
                parent = str(('ICD10',row.code[:l-1]))
                G.add_edge(parent, curr_node)
                non_leaf_codes.append(row.code[:l-1])
            code_mapping[curr_node] = curr_node
    # add icd9 entries to code mapping
    for i,row in icd9.iterrows():
        icd10code = icd9to10.icd10cm[icd9to10.icd9cm == row.code].iloc[0]
        code_mapping[str(('ICD9',row.code))] = str(('ICD10',icd10code))
    with open(code_mapping_file, 'wb') as f:
        pkl.dump(code_mapping, f)
    with open(code_graph_file, 'wb') as f:
        pkl.dump(G, f)
