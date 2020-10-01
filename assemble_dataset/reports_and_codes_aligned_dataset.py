import os
import random
from argparse import ArgumentParser
import pandas as pd
import pickle as pkl
from .reports_and_codes_dataset import ReportsToCodesAbstract, get_counts, to_file
from .dataset import Patient



class ReportsToCodesAligned(ReportsToCodesAbstract):
    @classmethod
    def get_datapoints(cls, patient_id, counts, reports=None, codes=None, code_mapping=None, code_graph=None):
        code_set = cls.code_set(codes, code_mapping=code_mapping)
        codes = codes[codes.patient_id==patient_id]
        patient = Patient(patient_id, reports=reports, codes=codes)
        reports = patient.compile_reports()
        reports = reports[reports.hadm_id == reports.hadm_id]
        reports.hadm_id = reports.hadm_id.astype('int')
        codes.hadm_id = codes.hadm_id.astype('int')
        reports = reports[reports.report_type=='Discharge summary']
        hadm_ids = set(reports.hadm_id)
        datapoints = []
        for hadm_id in hadm_ids:
            positive_targets = [code_mapping[str((row.code_type,row.code)).replace('.', '')]
                                for i,row in codes[codes.hadm_id==hadm_id][['code_type','code']].drop_duplicates().iterrows()
                                if str((row.code_type,row.code)).replace('.', '') in code_mapping.keys()
                                and code_mapping[str((row.code_type,row.code)).replace('.', '')] is not None]
            #positive_targets = list(cls.ancestors(code_graph, positive_targets))
            negative_targets = list(code_set.difference(set(positive_targets)))
            if len(positive_targets) == 0: continue
            datapoints.append([reports[reports.hadm_id==hadm_id].applymap(str).to_dict(), positive_targets+negative_targets, [1]*len(positive_targets)+[0]*len(negative_targets)])
        return datapoints
        """
        grouped_report_ids = pd.DataFrame({'date':reports.date, 'ids':reports.index}).groupby(pd.Grouper(key='date', freq='D')).aggregate(list)
        datapoints = []
        for date,report_ids in grouped_report_ids.iterrows():
            reports_bin = reports.loc[report_ids.ids]
            if len(report_ids.ids) == 0: continue
            codes = patient.get_codes(after_date=date, before_date=date+pd.to_timedelta('1 day'))
            positive_targets = [code_mapping[str((code_type, code)).replace('.', '')]
                                for code_type,code in codes
                                if str((code_type, code)).replace('.', '') in code_mapping.keys()
                                and code_mapping[str((code_type, code)).replace('.', '')] is not None]
            if len(positive_targets) == 0: continue
            positive_targets = list(cls.ancestors(code_graph, positive_targets))
            negative_targets = list(code_set.difference(set(positive_targets)))
            datapoints.append([reports_bin, positive_targets+negative_targets, [1]*len(positive_targets)+[0]*len(negative_targets)])
        return datapoints
        """

    @classmethod
    def columns(cls):
        return ['reports', 'targets', 'labels']


def main():
    parser = ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("dataset_name")
    parser.add_argument("--code_mapping")
    parser.add_argument("--code_graph")
    args = parser.parse_args()
    reports = pd.read_csv(os.path.join(args.folder, 'medical_reports.csv'), parse_dates=['date'])
    codes = pd.read_csv(os.path.join(args.folder, 'medical_codes.csv'), parse_dates=['date'])
    if args.code_mapping is not None:
        with open(args.code_mapping, 'rb') as f:
            code_mapping = pkl.load(f)
    else:
        code_mapping = None
    if args.code_graph is not None:
        with open(args.code_graph, 'rb') as f:
            code_graph = pkl.load(f)
    else:
        code_graph = None
    patient_ids = list(set(reports.patient_id))
    random.seed(0)
    random.shuffle(patient_ids)
    div1 = int(len(patient_ids)*.7)
    div2 = int(len(patient_ids)*.85)
    new_folder = os.path.join(args.folder, args.dataset_name)
    os.mkdir(new_folder)
    train_dataset = ReportsToCodesAligned.create(patient_ids[:div1], reports=reports, codes=codes, code_mapping=code_mapping, code_graph=code_graph)
    counts = get_counts(train_dataset)
    with open(os.path.join(new_folder, 'counts.pkl'), 'wb') as f:
        pkl.dump(counts, f)
    print(counts)
    threshold = 0
    useless_codes = set([k for k,v in counts.items() if v[0] < threshold or v[1] < threshold])
    print('useless codes:', useless_codes)
    usefull_codes = set(counts.keys()).difference(useless_codes)
    with open(os.path.join(new_folder, 'used_targets.txt'), 'w') as f:
        f.write(str(list(usefull_codes)))
    print('usefull codes:', usefull_codes)
    for k,v in code_mapping.items():
        if v in useless_codes:
            code_mapping[k] = None
    train_dataset = ReportsToCodesAligned.create(patient_ids[:div1], reports=reports, codes=codes, code_mapping=code_mapping, code_graph=code_graph)
    #train_dataset.to_json(os.path.join(new_folder, 'train.data'), orient='records', lines=True, compression='gzip', date_format="iso")
    to_file(train_dataset, os.path.join(new_folder, 'train.data'))
    val_dataset = ReportsToCodesAligned.create(patient_ids[div1:div2], reports=reports, codes=codes, code_mapping=code_mapping, code_graph=code_graph)
    #val_dataset.to_json(os.path.join(new_folder, 'val.data'), orient='records', lines=True, compression='gzip', date_format="iso")
    to_file(val_dataset, os.path.join(new_folder, 'val.data'))
    test_dataset = ReportsToCodesAligned.create(patient_ids[div2:], reports=reports, codes=codes, code_mapping=code_mapping, code_graph=code_graph)
    #test_dataset.to_json(os.path.join(new_folder, 'test.data'), orient='records', lines=True, compression='gzip', date_format="iso")
    to_file(test_dataset, os.path.join(new_folder, 'test.data'))
