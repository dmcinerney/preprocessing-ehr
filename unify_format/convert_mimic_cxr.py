import os
import pandas as pd
from tqdm import tqdm

data_folder = '/home/jered/Documents/data/alpha.physionet.org/files/mimic-cxr/2.0.0'

def main():
    df = pd.read_csv(os.path.join(data_folder, 'cxr-study-list.csv.gz'))
    text_columns = ['patient_id', 'date', 'report_type', 'description', 'text', 'from_table', 'other_info']
    datapoints = []
    import pdb; pdb.set_trace()
    for i,row in df.iterrows():
        text = open(os.path.join(data_folder, 'mimic-cxr-reports', row.path)).read()
        date = None # TODO: get date here
        datapoints.append([
            row.subject_id,
            date,
            "Radiology",
            None,
            text,
            'cxr-study-list.csv.gz,path:'+row.path,
            {},
        ])
