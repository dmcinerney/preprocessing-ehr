import os
from argparse import ArgumentParser
import re
import pandas as pd
from tqdm import tqdm
from random import shuffle
from assemble_dataset.patient import Patient

folder = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/'

def get_impression(report):
    results = re.findall('(%s:?[\s\n]*.+?)([$_])' % "IMPRESSION", report, flags=re.DOTALL)
    if len(results) > 0:
        return results[0][0]

def get_findings(report):
    results = re.findall('(%s:?[\s\n]*.+?)(\n\s*\n|[$_])' % "FINDINGS", report, flags=re.DOTALL)
    if len(results) > 0:
        return results[0][0]

def get_history(report):
    results = re.findall('(%s:?[\s\n]*.+?)(\n\s*\n|[$_])' % "HISTORY", report, flags=re.DOTALL)
    if len(results) > 0:
        return results[0][0]

def create_dataset(reports_df, patient_ids, filename):
    datapoints = []
    num_target_radiology_reports = 0
    num_with_impression = 0
    num_with_findings = 0
    num_with_history = 0
    pbar = tqdm(patient_ids, total=len(patient_ids))
    for patient_id in pbar:
        patient = Patient(patient_id, reports=reports_df)
        for i,row in patient.reports.iterrows():
            if row.report_type == 'Radiology':
                reports = patient.concatenate_reports(before_date=row.date-pd.to_timedelta('1 day'))
                if len(reports) == 0:
                    continue
                num_target_radiology_reports += 1
                impression = get_impression(row.text)
                if impression is not None:
                    num_with_impression += 1
                findings = get_findings(row.text)
                if findings is not None:
                    num_with_findings += 1
                history = get_history(row.text)
                if history is not None:
                    num_with_history += 1
                datapoints.append([reports, row.text, impression, findings, history])
            pbar.set_postfix({'num_radiology_reports':num_target_radiology_reports,
                              'num_impressions':num_with_impression,
                              'num_findings':num_with_findings,
                              'history':num_with_history})
    df = pd.DataFrame(datapoints, columns=['reports', 'radiology_report', 'impression', 'findings', 'history'])
#    df.to_json(filename, orient='records', lines=True, compression='gzip')
    df.to_csv(filename, compression='gzip', index=False)

def main():
    parser = ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    reports_df = pd.read_csv(os.path.join(args.folder, 'medical_reports.csv'), parse_dates=['date'])
    patient_ids = list(set(reports_df.patient_id))
    shuffle(patient_ids)
    div1 = int(len(patient_ids)*.7)
    div2 = int(len(patient_ids)*.85)
    new_folder = os.path.join(args.folder, 'reports_to_seq')
    os.mkdir(new_folder)
    create_dataset(reports_df, patient_ids[:div1], os.path.join(new_folder, 'train.data'))
    create_dataset(reports_df, patient_ids[div1:div2], os.path.join(new_folder, 'val.data'))
    create_dataset(reports_df, patient_ids[div2:], os.path.join(new_folder, 'test.data'))
