from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import pickle as pkl
import torch
from argparse import ArgumentParser
import pandas as pd
import pickle as pkl
from tqdm import tqdm

def get_codes(df, code_file=None):
    codes = set()
    for i,row in df.iterrows():
        for code in row.targets:
            codes.add(code)
    if code_file is not None:
        with open(code_file, 'wb') as f:
            pkl.dump(list(codes), f)
    return codes

class MedicalPredictionDataset(Dataset):
    # saves to files in a folder called name
    @classmethod
    def create(cls, reports, codes, code_metainfo=None, folder=None):
        patient_ids = set(list(reports.patient_id) + list(codes.patient_id))
        data = []
        code_mapping = code_metainfo.code_mapping if code_metainfo is not None else None
        for patient_id in tqdm(patient_ids, total=len(patient_ids)):
            data.extend(cls.get_datapoints(reports, codes, patient_id, code_mapping=code_mapping))
        df = pd.DataFrame(data, columns=cls.columns())
        dataset = cls(code_metainfo, df)
        if folder is not None:
            with open(os.path.join(folder, 'dataset.pkl'), 'wb') as datasetfile:
                pkl.dump(dataset, datasetfile)
        return dataset

    @classmethod
    def load_from_folder(cls, folder):
        with open(os.path.join(folder, 'dataset.pkl'), 'rb') as datasetfile:
            return pkl.load(datasetfile)

    @classmethod
    def columns(cls):
        raise NotImplementedError

    @classmethod
    def get_datapoints(cls, reports, codes, patient_id):
        raise NotImplementedError

    def __init__(self, code_metainfo, dataframe):
        self.code_metainfo = code_metainfo
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {column: self.df[column][int(index) if type(index) is torch.Tensor else index] for column in self.columns()}

class Patient:
    def __init__(self, reports, codes):
        self.reports = reports
        self.codes = codes

    def concatenate_reports(self, before_date=None):
        report_separator = '<report_separator>'
        time_separator = '<time_separator>'
        concatenated_reports = ''
        valid_rows = self.reports[self.reports.date < before_date]\
                     if before_date is not None else self.reports
        prev_date = None
        for i,row in valid_rows.iterrows():
            days = 0 if prev_date is None else (row.date-prev_date).days
            prev_date = row.date
            concatenated_reports += '\n\n'
            concatenated_reports += (' ' + time_separator)*days
            concatenated_reports += ' ' + report_separator
            concatenated_reports += ' ' + row.text
        return concatenated_reports

    def targets(self, code_mapping=None):
        # TODO: change this to using groupbys
        unique_codes = self.codes[['code_type', 'code']].drop_duplicates()
        targets = {}
        for i,row in unique_codes.iterrows():
            code = str((row.code_type, row.code)).replace('.','')
            if code_mapping is not None and code not in code_mapping.keys():
                continue
            code_rows = self.codes[(self.codes.code_type == row.code_type) & (self.codes.code == row.code)]
            if code_mapping is not None:
                targets[code_mapping[code]] = code_rows if code_mapping[code] not in targets.keys() else\
                                              pd.concat([targets[code_mapping[code]], code_rows], 0)
            else:
                targets[code] = code_rows
        targets = [(target,rows.sort_values('date')) for target,rows in targets.items()]
        # targets = sorted(targets, key=lambda t:-len(t[1]))
        return targets

class ReportsToCodes(MedicalPredictionDataset):
    @classmethod
    def get_datapoints(cls, reports, codes, patient_id, code_mapping=None, frequency_threshold=5):
        code_set = cls.code_set(codes, code_mapping=code_mapping)
        patient_reports = reports[reports.patient_id == patient_id].sort_values('date')
        patient_codes = codes[codes.patient_id == patient_id].sort_values('date')
        patient = Patient(patient_reports, patient_codes)
        target_list = patient.targets(code_mapping=code_mapping)
        targets = pd.DataFrame([[target, rows.date.iloc[0], len(rows)] for target,rows in target_list], columns=['target', 'date', 'frequency'])
        datapoints = []
        persistent_targets = targets[targets.frequency > frequency_threshold].sort_values('date')
        for i,target_row in persistent_targets.iterrows():
            target_date = target_row.date
            reports_text = patient.concatenate_reports(before_date=target_date-pd.to_timedelta('1 day'))
            if len(reports_text) == 0:
                continue
            negative_targets = list(code_set.difference(set(persistent_targets.target.tolist())))
            positive_targets = persistent_targets.target[persistent_targets.date >= target_date].tolist()
            datapoints.append([reports_text, positive_targets+negative_targets, [1]*len(positive_targets)+[0]*len(negative_targets)])
        return datapoints

    @classmethod
    def columns(cls):
        return ['reports', 'targets', 'labels']

    @classmethod
    def code_set(cls, codes, code_mapping=None):
        if code_mapping is None:
            unique_codes = codes[['code_type','code']].drop_duplicates()
            code_set = set(str((code_type, code)).replace('.','') for code_type, code in zip(unique_codes.code_type.tolist(), unique_codes.code.tolist()))
        else:
            code_set = code_mapping.values()
        return code_set

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    #reports = pd.read_csv(os.path.join(args.folder, 'medical_reports.csv'), parse_dates=['date'])
    #codes = pd.read_csv(os.path.join(args.folder, 'medical_codes.csv'), parse_dates=['date'])
    #dataset = ReportsToCodes.create(reports, codes, folder=args.folder)
    with open(os.path.join(args.folder, 'dataset.pkl'), 'rb') as f:
        dataset = pkl.load(f)
    div1 = int(len(dataset.df)*.7)
    div2 = int(len(dataset.df)*.85)
    dataset.df[:div1].to_json(os.path.join(args.folder, 'train_mimic.data'), orient='records', lines=True, compression='gzip')
    dataset.df[div1:div2].to_json(os.path.join(args.folder, 'val_mimic.data'), orient='records', lines=True, compression='gzip')
    dataset.df[div2:].to_json(os.path.join(args.folder, 'test_mimic.data'), orient='records', lines=True, compression='gzip')
    get_codes(dataset.df, code_file=os.path.join(args.folder, 'codes.pkl'))
