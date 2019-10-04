import pandas as pd
import pickle as pkl
import networkx as nx

code_mapping_file = '/home/jered/Documents/data/icd_codes/code_mapping.pkl'
code_graph_file = '/home/jered/Documents/data/icd_codes/code_graph.pkl'
icd_and_readmission_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/icd_and_readmission/icd_and_readmission.data'

def main():
    with open(code_mapping_file, 'rb') as f:
        code_mapping = pkl.load(f)
    with open(code_graph_file, 'rb') as f:
        code_graph = pkl.load(f)
    df = pd.read_json(icd_and_readmission_file, lines=True, compression='gzip')
    for n,d in code_graph.in_degree():
        if d == 0:
            nodes = list(nx.algorithms.dag.descendants(code_graph, n)) + [n]
            columns = [name for name in df.columns if name in code_mapping.keys() and code_mapping[name] in nodes]
            if len(columns) > 0:
                category_present = df[columns].fillna(0).sum(1) >= 1
                import pdb; pdb.set_trace()
