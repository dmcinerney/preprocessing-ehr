import pickle as pkl

class CodeMetaInfo:
    @classmethod
    def create(cls, code_mapping_file, code_graph_file):
        with open(code_mapping_file, 'rb') as file:
            code_mapping = pkl.load(file)
        with open(code_graph_file, 'rb') as file:
            code_graph = pkl.load(file)
        return cls(code_mapping, code_graph)

    def __init__(self, code_mapping, code_graph):
        self.code_mapping = code_mapping
        self.code_graph = code_graph

