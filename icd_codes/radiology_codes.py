import pandas as pd
import pickle as pkl
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_postorder_nodes
from networkx.classes.function import set_node_attributes
from tqdm import tqdm

outline_file = 'icd_codes/radiology_codes_outline.txt'
original_code_mapping_file = '/home/jered/Documents/data/icd_codes/code_mapping.pkl'
original_code_graph_file = '/home/jered/Documents/data/icd_codes/code_graph.pkl'
new_code_mapping_file = '/home/jered/Documents/data/icd_codes/code_mapping_radiology.pkl'
new_code_graph_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology.pkl'

def create_graph(filename):
    G = nx.DiGraph()
    prev_tab_level = None
    start_node = '<start_node>'
    current_nodes_at_each_depth = []
    with open(filename, 'r') as file:
        for i,line in enumerate(file):
            if len(line.strip()) == 0: continue
            num_spaces = len(line) - len(line.lstrip(' '))
            new_tab_level = num_spaces//4
            tab_difference = new_tab_level - prev_tab_level if prev_tab_level is not None else 0
            prev_tab_level = new_tab_level
            # print(i+1, new_tab_level, tab_difference)
            if tab_difference > 1:
                raise Exception
            if len(current_nodes_at_each_depth) < new_tab_level + 1:
                current_nodes_at_each_depth.append(None)
            linesplit = line.strip().split()
            indices = [i for i,token in enumerate(linesplit) if any(char.isdigit() for char in token)]
            if len(indices) > 0:
                nodename = linesplit[indices[0]]
                del linesplit[indices[0]]
                description = ' '.join(linesplit)
                G.add_node(nodename, description=description)
            else:
                nodename = line.strip()
                G.add_node(nodename)
            current_nodes_at_each_depth[new_tab_level] = nodename
            if new_tab_level == 0:
                G.add_edge(start_node, nodename)
            else:
                G.add_edge(current_nodes_at_each_depth[new_tab_level-1], nodename)
    return G

def splitints(string):
    letters = ''
    numbers = ''
    for char in string:
        if char.isdigit():
            numbers += char
        else:
            letters += char
    return letters, numbers

# really hacky
def getids(nodename):
    if not any(char.isdigit() for char in nodename):
        raise Exception
    nodename = nodename.replace('.','')
    if '-' in nodename:
        start, stop = nodename.split('-')
        startl, startn = splitints(start)
        stopl, stopn = splitints(stop)
        if startl != stopl or (startl+startn) != start or (stopl+stopn) != stop:
            raise Exception
        return [startl+str(i)[1:] for i in range(int('1'+startn), int('1'+stopn)+1)]
    else:
        return [nodename]

def create_icd_mapping_and_modify_graph(G, original_G, original_code_mapping, only_leaf_codes=True):
    new_code_mapping = {}
    node_codes = {}
    df = pd.DataFrame(list(original_code_mapping.keys()))
    postorder = dfs_postorder_nodes(G, source='<start_node>')
    for n in tqdm(postorder, total=len(G.nodes)):
        node_codes[n] = {'codes':set()}
        if only_leaf_codes and len(G[n]) != 0:
            continue
        if not any(char.isdigit() for char in n):
            continue
        ids = getids(n)
        codes9 = []
        codes10 = []
        for i in ids:
            codes9 += list(df[0][df[0].str.startswith(str(('ICD9', i))[:-2])])
            codes10 += list(df[0][df[0].str.startswith(str(('ICD10', i))[:-2])])
        if len(codes10) > 0 and len(codes10) > len(codes9):
            codes = codes10
        elif len(codes9) > 0:
            codes = codes9
        else:
            continue
        for code in codes:
            code = original_code_mapping[code]
            new_code_mapping[code] = n
            node_codes[n]['codes'].add(code)
            for descendant in nx.descendants(original_G, code):
                new_code_mapping[descendant] = n
                node_codes[n]['codes'].add(descendant)
    set_node_attributes(G, node_codes)
    import pdb; pdb.set_trace()
    for k,v in original_code_mapping.items():
        if v not in new_code_mapping.keys():
            new_code_mapping[k] = None
        else:
            new_code_mapping[k] = new_code_mapping[v]
    return new_code_mapping

def main():
    with open(original_code_mapping_file, 'rb') as f:
        original_code_mapping = pkl.load(f)
    with open(original_code_graph_file, 'rb') as f:
        original_G = pkl.load(f)
    G = create_graph(outline_file)
    new_code_mapping = create_icd_mapping_and_modify_graph(G, original_G, original_code_mapping)
    with open(new_code_mapping_file, 'wb') as f:
        pkl.dump(new_code_mapping, f)
    with open(new_code_graph_file, 'wb') as f:
        pkl.dump(G, f)
