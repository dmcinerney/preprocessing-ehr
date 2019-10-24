import pickle as pkl
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_postorder_nodes
from networkx.classes.function import set_node_attributes

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
            nodename = linesplit[0]
            current_nodes_at_each_depth[new_tab_level] = nodename
            if len(linesplit) > 1:
                description = ' '.join(linesplit[1:])
                G.add_node(nodename, description=description)
            else:
                G.add_node(nodename)
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
    if any(char.isdigit() for char in nodename):
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
    for n in dfs_postorder_nodes(G, source='<start_node>'):
        node_codes[n] = {'codes':[]}
        if only_leaf_codes and len(G[n]) != 0:
            continue
        if any(char.isdigit() for char in n):
            ids = getids(n)
            for i in ids:
                code = str(('ICD10', i))
                if code not in original_G.nodes:
                    break
                new_code_mapping[code] = n
                node_codes[n]['codes'].append(code)
                for descendant in nx.descendants(original_G, code):
                    new_code_mapping[descendant] = n
                    node_codes[n]['codes'].append(descendant)
    set_node_attributes(G, node_codes)
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
