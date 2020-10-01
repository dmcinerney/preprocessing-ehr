import os
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import random
from utils import to_file
from assemble_dataset.dataset import MedicalDataset, Patient
import subprocess
import queue

def get_token_spans(tokenplace, reporttext):
    line, index = tokenplace.split(':')
    line, index = int(line), int(index)
    splittext = reporttext.split('\n')
    splitline = splittext[line-1].split(' ')
    endindex = len('\n'.join(splittext[:line-1] + [
        ' '.join(splitline[:index+1])
    ]))
    startindex = endindex - len(splitline[index])
    return startindex, endindex

def get_char_spans(starttoken, endtoken, reporttext):
    start, _ = get_token_spans(starttoken, reporttext)
    _, end = get_token_spans(endtoken, reporttext)
    return start, end

def add_entities_wrapper(entity_path):
    def add_entities(row):
        entity_spans = []
        entity_types = []
        skipped_entities = []
        with open(os.path.join(entity_path, 'report%i.con' % row.name), 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0: continue
                span, entity_type = line.split('||')
                spansplit = span.split(' ')
                starttoken, endtoken = spansplit[-2:]
                try:
                    span = get_char_spans(starttoken, endtoken, row.text)
                except Exception as e:
                    entity_string = ' '.join(spansplit[:-2])[3:-1]
                    skipped_entities.append([[starttoken, endtoken], entity_string, e])
                    continue
                entity_spans.append(span)
                entity_types.append(entity_type[3:-1])
        row['entity_spans'] = entity_spans
        row['entity_types'] = entity_types
        row['skipped_entities'] = skipped_entities
        return row
    return add_entities

class ReportsEntityLanguageModeling(MedicalDataset):
    @classmethod
    def get_datapoints(cls, patient_id, counts, reports, entity_path):
        patient = Patient(patient_id, reports=reports)
        reports = patient.compile_reports()
        reports = reports.apply(add_entities_wrapper(entity_path), axis=1)
        counts["reports"] += len(reports)
        counts["entities"] += reports.entity_types.apply(len).sum()
        counts["skipped_entities"] += reports.skipped_entities.apply(len).sum()
        #if reports.skipped_entities.apply(len).sum() > 0:
        #    import pdb; pdb.set_trace()
        return [[reports.applymap(str).to_dict()]]

    @classmethod
    def init_count_dict(cls):
        return {**super(ReportsEntityLanguageModeling, cls).init_count_dict(),
                'reports':0,
                'entities':0,
                'skipped_entities':0}

    @classmethod
    def columns(cls):
        return ["reports"]

def create_reports_text_folder(reports, folder):
    # os.mkdir(folder)
    for i,row in tqdm(reports.iterrows(), total=len(reports)):
        file = os.path.join(folder, "report%i.txt" % i)
        with open(file, 'w') as f:
            f.write(row.text)

def tag_entities(input_folder, output_folder, cliner_folder):
    # process = subprocess.Popen(["python", "cliner", "predict", "--txt", os.path.join(input_folder, "report[0-9]*.txt"),
    #                             "--out", output_folder, "--format", "i2b2", "--model", "models/silver.crf"],
    #                            cwd=cliner_folder)
    # process.wait()
    # files = list(os.listdir(input_folder))
    regexes = [os.path.join(input_folder, "report[0-9]*0.txt"),
               os.path.join(input_folder, "report[0-9]*1.txt"),
               os.path.join(input_folder, "report[0-9]*2.txt"),
               os.path.join(input_folder, "report[0-9]*3.txt"),
               os.path.join(input_folder, "report[0-9]*4.txt"),
               os.path.join(input_folder, "report[0-9]*5.txt"),
               os.path.join(input_folder, "report[0-9]*6.txt"),
               os.path.join(input_folder, "report[0-9]*7.txt"),
               os.path.join(input_folder, "report[0-9]*8.txt"),
               os.path.join(input_folder, "report[0-9]*9.txt"),]
    processes = queue.Queue()
    for regex in regexes:
        process = subprocess.Popen(["python", "cliner", "predict", "--txt", regex,
                                    "--out", output_folder, "--format", "i2b2", "--model", "models/silver.crf"],
                                   cwd=cliner_folder)
        processes.put(process)
    while len(processes) > 0:
        processes.get().wait()

def main():
    parser = ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("dataset_name")
    parser.add_argument("cliner_folder")
    args = parser.parse_args()
    reports = pd.read_csv(os.path.join(args.folder, 'medical_reports.csv'), parse_dates=['date'])
    # create_report_text_folder(reports, os.path.join(args.folder, "report_text"))
    # tag_entities(os.path.join(args.folder, "report_text"), os.path.join(args.folder, "report_entities"), args.cliner_folder)
    patient_ids = list(set(reports.patient_id))[:100]
    # create_report_text_folder(reports[reports.patient_id.isin(patient_ids)], os.path.join(args.folder, "report_text"))
    # tag_entities(os.path.join(args.folder, "report_text"), os.path.join(args.folder, "report_entities"), args.cliner_folder)
    random.seed(0)
    random.shuffle(patient_ids)
    div1 = int(len(patient_ids)*.7)
    div2 = int(len(patient_ids)*.85)
    new_folder = os.path.join(args.folder, args.dataset_name)
    os.mkdir(new_folder)
    train_dataset = ReportsEntityLanguageModeling.create(patient_ids[:div1], reports=reports, entity_path=os.path.join(args.folder, "report_entities"))
    to_file(train_dataset, os.path.join(new_folder, 'train.data'))
    val_dataset = ReportsEntityLanguageModeling.create(patient_ids[div1:div2], reports=reports, entity_path=os.path.join(args.folder, "report_entities"))
    to_file(val_dataset, os.path.join(new_folder, 'val.data'))
    test_dataset = ReportsEntityLanguageModeling.create(patient_ids[div2:], reports=reports, entity_path=os.path.join(args.folder, "report_entities"))
    to_file(test_dataset, os.path.join(new_folder, 'test.data'))
