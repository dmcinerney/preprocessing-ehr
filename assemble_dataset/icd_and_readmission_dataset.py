import os
import gzip
import json
import pandas as pd
from tqdm import tqdm
from patient import Patient

codes_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/medical_codes.csv'
admissions_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/admissions.csv'
output_folder = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/icd_and_readmission'

if __name__ == '__main__':
    codes_df = pd.read_csv(codes_file, parse_dates=['date'])
    admissions_df = pd.read_csv(admissions_file, parse_dates=['date_start', 'date_end'])
    patient_ids = list(set(list(codes_df.patient_id)))
    with gzip.open(os.path.join(output_folder, 'icd_and_readmission.data'), 'w') as f:
        for patient_id in tqdm(patient_ids):
            patient = Patient(patient_id, codes=codes_df, admissions=admissions_df)
            for i in range(len(patient.admissions)):
                codes = patient.get_codes(after_date=patient.admissions.iloc[i].date_start,
                                          before_date=patient.admissions.iloc[i].date_end)
                json_obj = json.dumps(
                    {'patient_id':patient_id, 'admission_number':i,
                     'readmission_label':(1 if i < len(patient.admissions)-1 else 0),
                     **{str(code):1 for code in codes}})
                f.write(bytes(json_obj+'\n', encoding='utf-8'))
