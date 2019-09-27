import pandas as pd
import numpy as np
import os
import pickle as pkl
from argparse import ArgumentParser
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from patient import Patient

def get_codes(df, code_file=None):
    codes = set()
    for i,row in df.iterrows():
        for code in row.targets:
            codes.add(code)
    if code_file is not None:
        with open(code_file, 'wb') as f:
            pkl.dump(list(codes), f)
    return codes

class MedicalPredictionDataset:
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

class ReportsToCodes(MedicalPredictionDataset):
    @classmethod
    def get_datapoints(cls, reports, codes, patient_id, code_mapping=None, frequency_threshold=5):
        code_set = cls.code_set(codes, code_mapping=code_mapping)
        patient = Patient(patient_id, reports=reports, codes=codes)
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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    reports = pd.read_csv(os.path.join(args.folder, 'medical_reports.csv'), parse_dates=['date'])
    codes = pd.read_csv(os.path.join(args.folder, 'medical_codes.csv'), parse_dates=['date'])
    dataset = ReportsToCodes.create(reports, codes, folder=args.folder)
    with open(os.path.join(args.folder, 'dataset.pkl'), 'rb') as f:
        dataset = pkl.load(f)
    div1 = int(len(dataset.df)*.7)
    div2 = int(len(dataset.df)*.85)
    new_folder = os.path.join(args.folder, 'preprocessed', 'reports_and_codes')
    os.mkdir(new_folder)
    dataset.df[:div1].to_json(os.path.join(new_folder, 'train_mimic.data'), orient='records', lines=True, compression='gzip')
    dataset.df[div1:div2].to_json(os.path.join(new_folder, 'val_mimic.data'), orient='records', lines=True, compression='gzip')
    dataset.df[div2:].to_json(os.path.join(new_folder, 'test_mimic.data'), orient='records', lines=True, compression='gzip')
    get_codes(dataset.df, code_file=os.path.join(new_folder, 'codes.pkl'))
