from argparse import ArgumentParser
from patient import Patient
from random import shuffle
import pandas as pd
import os
from tqdm import tqdm



def create_dataset(patient_ids, new_folder, reports):
    print("creating "+new_folder)
    os.mkdir(new_folder)
    for i,patient_id in tqdm(enumerate(patient_ids), total=len(patient_ids)):
        report = Patient(patient_id, reports=reports).concatenate_reports()
        report += '\n\n[PAD]'*509
        with open(os.path.join(new_folder, 'patient_%i.txt' % patient_id), 'w') as f:
            f.write(report)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    reports = pd.read_csv(os.path.join(args.folder, 'medical_reports.csv'), parse_dates=['date'])
    patient_ids = list(set(list(reports.patient_id)))
    shuffle(patient_ids)
    new_folder = os.path.join(args.folder, 'reports_txt')
    os.mkdir(new_folder)
    div1 = int(len(patient_ids)*.7)
    div2 = int(len(patient_ids)*.85)
    create_dataset(patient_ids[:div1], os.path.join(new_folder, 'train'), reports)
    create_dataset(patient_ids[div1:div2], os.path.join(new_folder, 'val'), reports)
    create_dataset(patient_ids[div2:], os.path.join(new_folder, 'test'), reports)
