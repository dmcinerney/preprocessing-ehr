import pandas as pd
import numpy as np
import os
import pickle as pkl
from argparse import ArgumentParser
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from random import shuffle
from assemble_dataset.patient import Patient


class MedicalPredictionDataset:
    # saves to files in a folder called name
    @classmethod
    def create(cls, patient_ids, reports, codes, code_mapping=None):
        data = []
        iterator = tqdm(patient_ids, total=len(patient_ids))
        total_skipped, total_retrieved, total_num_datapoints = 0, 0, 0
        for patient_id in iterator:
            datapoints, skipped, retrieved = cls.get_datapoints(reports, codes, patient_id, code_mapping=code_mapping)
            data.extend(datapoints)
            total_skipped += skipped
            total_retrieved += retrieved
            total_num_datapoints += len(datapoints)
            iterator.set_postfix({'skipped':total_skipped,
                                  'retrieved':total_retrieved,
                                  'num_datapoints':total_num_datapoints})
        return pd.DataFrame(data, columns=cls.columns())

    @classmethod
    def columns(cls):
        raise NotImplementedError

    @classmethod
    def get_datapoints(cls, reports, codes, patient_id):
        raise NotImplementedError

class ReportsToCodes(MedicalPredictionDataset):
    @classmethod
    def get_datapoints(cls, reports, codes, patient_id, code_mapping=None, frequency_threshold=3):
        code_set = cls.code_set(codes, code_mapping=code_mapping)
        patient = Patient(patient_id, reports=reports, codes=codes)
        target_list, skipped = patient.targets(code_mapping=code_mapping)
        targets = pd.DataFrame([[target, rows.date.iloc[0], len(rows)] for target,rows in target_list], columns=['target', 'date', 'frequency'])
        retreived = len(targets)
        datapoints = []
        persistent_targets = targets[targets.frequency >= frequency_threshold].sort_values('date')
        prev_patient_report_index = None
        for i,target_row in persistent_targets.iterrows():
            target_date = target_row.date
            patient_reports = patient.compile_reports(before_date=target_date-pd.to_timedelta('1 day'))
            if len(patient_reports) == 0 or patient_reports.index[-1] == prev_patient_report_index:
                continue
            prev_patient_report_index = patient_reports.index[-1]
            negative_targets = list(code_set.difference(set(persistent_targets.target.tolist())))
            positive_targets = persistent_targets.target[persistent_targets.date >= target_date].tolist()
            datapoints.append([patient_reports, positive_targets+negative_targets, [1]*len(positive_targets)+[0]*len(negative_targets)])
        return datapoints, skipped, retreived

    @classmethod
    def columns(cls):
        return ['reports', 'targets', 'labels']

    @classmethod
    def code_set(cls, codes, code_mapping=None):
        if code_mapping is None:
            unique_codes = codes[['code_type','code']].drop_duplicates()
            code_set = set(str((code_type, code)).replace('.','') for code_type, code in zip(unique_codes.code_type.tolist(), unique_codes.code.tolist()))
        else:
            code_set = set(code_mapping.values()).difference(set([None]))
        return code_set

def get_counts(dataset):
    print("getting counts")
    counts = {}
    for i,row in tqdm(dataset.iterrows(), total=len(dataset)):
        for j in range(len(row.targets)):
            key = row.targets[j]
            if key not in counts.keys():
                counts[key] = [0, 0]
            counts[key][row.labels[j]] += 1
    return counts

def main():
    parser = ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--code_mapping")
    args = parser.parse_args()
    reports = pd.read_csv(os.path.join(args.folder, 'medical_reports.csv'), parse_dates=['date'])
    codes = pd.read_csv(os.path.join(args.folder, 'medical_codes.csv'), parse_dates=['date'])
    if args.code_mapping is not None:
        with open(args.code_mapping, 'rb') as f:
            code_mapping = pkl.load(f)
    else:
        code_mapping = None
    patient_ids = list(set(reports.patient_id))
    shuffle(patient_ids)
    div1 = int(len(patient_ids)*.7)
    div2 = int(len(patient_ids)*.85)
    new_folder = os.path.join(args.folder, 'reports_and_codes')
    os.mkdir(new_folder)
    train_dataset = ReportsToCodes.create(patient_ids[:div1], reports, codes, code_mapping=code_mapping)
    counts = get_counts(train_dataset)
    with open(os.path.join(new_folder, 'counts.pkl'), 'wb') as f:
        pkl.dump(counts, f)
    print(counts)
    threshold = 50
    useless_codes = set([k for k,v in counts.items() if v[0] < threshold or v[1] < threshold])
    print('useless codes:', useless_codes)
    print('usefull codes:', set(counts.keys()).difference(useless_codes))
    for k,v in code_mapping.items():
        if v in useless_codes:
            code_mapping[k] = None
    train_dataset = ReportsToCodes.create(patient_ids[:div1], reports, codes, code_mapping=code_mapping)
    train_dataset.to_json(os.path.join(new_folder, 'train.data'), orient='records', lines=True, compression='gzip')
    val_dataset = ReportsToCodes.create(patient_ids[div1:div2], reports, codes, code_mapping=code_mapping)
    val_dataset.to_json(os.path.join(new_folder, 'val.data'), orient='records', lines=True, compression='gzip')
    test_dataset = ReportsToCodes.create(patient_ids[div2:], reports, codes, code_mapping=code_mapping)
    test_dataset.to_json(os.path.join(new_folder, 'test.data'), orient='records', lines=True, compression='gzip')
