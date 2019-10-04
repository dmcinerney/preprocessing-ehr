import os
from argparse import ArgumentParser
import re
import pandas as pd
from tqdm import tqdm
from assemble_dataset.patient import Patient

folder = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/'

section_title = 'IMPRESSION'
regex = '(%s:?[\s\n]*.+?)(\n\s*\n|$)' % section_title

def main():
    parser = ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    reports_df = pd.read_csv(os.path.join(args.folder, 'medical_reports.csv'), parse_dates=['date'])
    patient_ids = list(set(reports_df.patient_id))
    datapoints = []
    for patient_id in tqdm(patient_ids, total=len(patient_ids)):
        patient = Patient(patient_id, reports=reports_df)
        for i,row in patient.reports.iterrows():
            import pdb; pdb.set_trace()
            if row.from_table == '':
                section = re.findall(regex, row.text, flags=re.DOTALL)[0][0]
                reports = patient.concatenate_reports(before_date=row.date-pd.to_datetime('1 day'))
                import pdb; pdb.set_trace()
                datapoints.append([reports, section])
    pd.DataFrame(datapoints, columns=['reports', 'target_text'])
    import pdb; pdb.set_trace()
