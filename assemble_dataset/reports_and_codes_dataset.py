import copy
import pandas as pd
import numpy as np
import os
import pickle as pkl
from argparse import ArgumentParser
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import random
from assemble_dataset.patient import Patient


class MedicalPredictionDataset:
    # saves to files in a folder called name
    @classmethod
    def create(cls, patient_ids, reports, codes, code_mapping=None, code_graph=None):
        data = []
        iterator = tqdm(patient_ids, total=len(patient_ids))
        counts = cls.init_count_dict()
        for patient_id in iterator:
            datapoints = cls.get_datapoints(reports, codes, patient_id, counts=counts, code_mapping=code_mapping, code_graph=code_graph)
            data.extend(datapoints)
            counts['num_datapoints'] += len(datapoints)
            iterator.set_postfix(counts)
        return pd.DataFrame(data, columns=cls.columns())

    @classmethod
    def update_counts(cls, counts, k, v):
        if counts is not None:
            counts[k] += v

    @classmethod
    def init_count_dict(cls):
        return {'num_datapoints':0}

    @classmethod
    def columns(cls):
        raise NotImplementedError

    @classmethod
    def get_datapoints(cls, reports, codes, patient_id, counts=None):
        raise NotImplementedError

class ReportsToCodesAbstract(MedicalPredictionDataset):
    @classmethod
    def ancestors(cls, graph, nodes, stop_nodes=set()):
        node_stack = copy.deepcopy(nodes)
        new_nodes = set()
        while len(node_stack) > 0:
            node = node_stack.pop()
            if node in stop_nodes: continue # don't add stop nodes
            if node in new_nodes: continue # don't add nodes already there
            in_degree = graph.in_degree(node)
            if in_degree == 0: continue # don't add the start node
            elif in_degree > 1: raise Exception # shouldn't have any nodes with more than one parent
            new_nodes.add(node)
            node_stack.extend(list(graph.predecessors(node)))
        return list(new_nodes)

    @classmethod
    def code_set(cls, codes, code_mapping=None):
        if code_mapping is None:
            unique_codes = codes[['code_type','code']].drop_duplicates()
            code_set = set(str((code_type, code)).replace('.','') for code_type, code in zip(unique_codes.code_type.tolist(), unique_codes.code.tolist()))
        else:
            code_set = set(code_mapping.values()).difference(set([None]))
        return code_set

    @classmethod
    def columns(cls):
        return ['reports', 'future_reports', 'targets', 'labels', 'previous_targets']


class ReportsToCodes(ReportsToCodesAbstract):
    @classmethod
    def get_datapoints(cls, reports, codes, patient_id, counts=None, code_mapping=None, code_graph=None, frequency_threshold=2):
        code_set = cls.code_set(codes, code_mapping=code_mapping)
        patient = Patient(patient_id, reports=reports, codes=codes)
        target_list, skipped = patient.targets(code_mapping=code_mapping)
        target_dates = {}
        for leaf_target,rows in target_list:
            for target in (cls.ancestors(code_graph, [leaf_target]) if code_graph is not None else [leaf_target]):
                if target not in target_dates.keys():
                    target_dates[target] = []
                target_dates[target].extend([row.date for i,row in rows.iterrows()])
        target_dates = {target:pd.Series(sorted(list(set(dates)))) for target,dates in target_dates.items()}
        cls.update_counts(counts, 'skipped', skipped)
        targets = pd.DataFrame([[target, rows.date.iloc[0], len(rows)] for target,rows in target_list], columns=['target', 'date', 'frequency']).sort_values('date')
        cls.update_counts(counts, 'retreived', len(targets))
        persistent_targets = targets[targets.frequency >= frequency_threshold]
        radiology_reports = patient.reports[patient.reports.report_type == "Radiology"]
        if len(radiology_reports) == 0:
            cls.update_counts(counts, 'no_radiology_reports', 1)
            return []
        datapoints = []
        #for i,row in radiology_reports.iterrows():
        #    target_date = row.date
        prev_target_date_index = -1
        for i,row in persistent_targets.iterrows():
            past_radiology_report_dates = radiology_reports.date[radiology_reports.date <= row.date-pd.to_timedelta('1 day')]
            if len(past_radiology_report_dates) == 0:
                cls.update_counts(counts, 'no_past_reports', 1)
                continue
            target_date = past_radiology_report_dates.iloc[-1]
            if prev_target_date_index == past_radiology_report_dates.index[-1]:
                cls.update_counts(counts, 'same_target_date', 1)
                continue
            prev_target_date_index = past_radiology_report_dates.index[-1]
            if target_date+pd.to_timedelta(1, unit='Y') < row.date:
                cls.update_counts(counts, 'no_recent_reports', 1)
                continue
            past_reports = patient.compile_reports(before_date=target_date)
            if len(past_reports) == 0:
                cls.update_counts(counts, 'no_past_reports', 1)
                continue
            future_reports = patient.compile_reports(after_date=target_date, before_date=target_date+pd.to_timedelta(1, unit='Y'))
            if len(future_reports) == 0:
                cls.update_counts(counts, 'no_future_reports', 1)
                continue
            positive_targets = persistent_targets[(persistent_targets.date >= target_date)\
                                                & (persistent_targets.date < target_date+pd.to_timedelta(1, unit='Y'))].target.tolist()
            if code_graph is not None:
                positive_targets = cls.ancestors(code_graph, positive_targets)
            # TODO: check that this should be zero
            if len(positive_targets) == 0:
                cls.update_counts(counts, 'no_pos_targets', 1)
                continue
            not_negative_nodes = set(cls.ancestors(code_graph, persistent_targets.target.tolist())
                                      if code_graph is not None else persistent_targets.target.tolist())
            negative_targets = list(cls.ancestors(code_graph, list(code_set), stop_nodes=not_negative_nodes)
                                    if code_graph is not None else code_set.difference(not_negative_nodes))
            previous_targets = {target:dates[(dates < target_date)\
                                           & (dates >= past_radiology_report_dates.iloc[0])].apply(str).to_list() for target,dates in target_dates.items()}
            previous_targets = {target:dates for target,dates in previous_targets.items() if len(dates) > 0}
            datapoints.append([
                past_reports.applymap(str).to_dict(),
                future_reports.applymap(str).to_dict(),
                positive_targets+negative_targets,
                [1]*len(positive_targets)+[0]*len(negative_targets),
                previous_targets])
        return datapoints

    @classmethod
    def init_count_dict(cls):
        return {**super(ReportsToCodes, cls).init_count_dict(),
                'skipped':0,
                'retreived':0,
                'no_radiology_reports':0,
                'no_past_reports':0,
                'same_target_date':0,
                'no_recent_reports':0,
                'no_future_reports':0,
                'no_pos_targets':0,}


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


def to_file(df, file, blocksize=1000):
    print("saving to "+file)
    pbar = tqdm(total=len(df))
    for i in range(0, len(df), blocksize):
        df[i:i+blocksize].to_csv(file, index=False, header=i==0, mode='a', compression='gzip')
        #df[i:i+blocksize].to_json(file, orient='records', lines=True, compression='gzip', date_format="iso", mode='a')
        pbar.update(n=len(df[i:i+blocksize]))

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
    patient_ids = list(set(reports.patient_id))[:100]
    random.seed(0)
    random.shuffle(patient_ids)
    div1 = int(len(patient_ids)*.7)
    div2 = int(len(patient_ids)*.85)
    new_folder = os.path.join(args.folder, args.dataset_name)
    os.mkdir(new_folder)
    train_dataset = ReportsToCodes.create(patient_ids[:div1], reports, codes, code_mapping=code_mapping, code_graph=code_graph)
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
    train_dataset = ReportsToCodes.create(patient_ids[:div1], reports, codes, code_mapping=code_mapping, code_graph=code_graph)
    #train_dataset.to_json(os.path.join(new_folder, 'train.data'), orient='records', lines=True, compression='gzip', date_format="iso")
    to_file(train_dataset, os.path.join(new_folder, 'train.data'))
    val_dataset = ReportsToCodes.create(patient_ids[div1:div2], reports, codes, code_mapping=code_mapping, code_graph=code_graph)
    #val_dataset.to_json(os.path.join(new_folder, 'val.data'), orient='records', lines=True, compression='gzip', date_format="iso")
    to_file(val_dataset, os.path.join(new_folder, 'val.data'))
    test_dataset = ReportsToCodes.create(patient_ids[div2:], reports, codes, code_mapping=code_mapping, code_graph=code_graph)
    #test_dataset.to_json(os.path.join(new_folder, 'test.data'), orient='records', lines=True, compression='gzip', date_format="iso")
    to_file(test_dataset, os.path.join(new_folder, 'test.data'))
