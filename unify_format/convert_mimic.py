import os
import pandas as pd

data_folder = "/home/jered/Documents/data/mimic-iii-clinical-database-1.4"
new_data_folder = "/home/jered/Documents/data/mimic_processed"

def create_report_csv(data_folder, new_data_folder):
    print("reading in reports")
    text_df = pd.read_csv(os.path.join(data_folder, "NOTEEVENTS.csv.gz"), low_memory=False)
    print("iterating through reports")
    text_rows = []
    text_columns = ['patient_id', 'date', 'report_type', 'description', 'text', 'from_table']
    for i,row in text_df.iterrows():
        text_rows.append([
            row.SUBJECT_ID,
            pd.to_datetime(row.CHARTDATE),
            row.CATEGORY,
            row.DESCRIPTION,
            row.TEXT,
            "NOTEEVENTS"
        ])
        if (i+1) % 1000 == 0:
            print(i+1, '/', len(text_df))
    pd.DataFrame(text_rows, columns=text_columns).to_csv(os.path.join(new_data_folder, 'medical_reports.csv'), index=False)

def create_code_csv(data_folder, new_data_folder):
    print("reading in codes")
    code_df = pd.read_csv(os.path.join(data_folder, "DIAGNOSES_ICD.csv.gz"), low_memory=False)
    adm_df = pd.read_csv(os.path.join(data_folder, "ADMISSIONS.csv.gz"), low_memory=False)
    code_names_df = pd.read_csv(os.path.join(data_folder, "D_ICD_DIAGNOSES.csv.gz"), low_memory=False)
    print("iterating through codes")
    code_rows = []
    code_columns = ['patient_id', 'date', 'flag', 'name', 'code_type', 'code', 'from_table']
    for i,row in code_df.iterrows():
        if row.ICD9_CODE == row.ICD9_CODE:
            icd_code_rows = code_names_df.SHORT_TITLE[code_names_df.ICD9_CODE.str.match(pat='^0*'+row.ICD9_CODE+'\d*$')]]
            code_rows.append([
                row.SUBJECT_ID,
                pd.to_datetime(adm_df.DISCHTIME[adm_df.HADM_ID==row.HADM_ID].iloc[0]),
                row.SEQ_NUM,
                [short_title for short_title in icd_code_rows.SHORT_TITLE,
                "ICD9",
#                [icd_code for icd_code in icd_code_rows.ICD9_CODE][0], # TODO: don't just select the first one
                row.ICD9_CODE,
                ["DIAGNOSES_ICD.csv.gz","ADMISSIONS.csv.gz","D_ICD_DIAGNOSES.csv.gz"]
            ])
        if (i+1) % 1000 == 0:
            print(i+1, '/', len(code_df))
    pd.DataFrame(code_rows, columns=code_columns).to_csv(os.path.join(new_data_folder, 'medical_codes.csv'), index=False)

"""
def create_code_csv(data_folder, new_data_folder):
#    code_df = pd.read_csv(os.path.join(data_folder, "DIAGNOSES_ICD.csv.gz"), low_memory=False)
    count = 0
    code_columns = ['patient_id', 'date', 'flag', 'name', 'code_type', 'code', 'from_table']
    print("iterating through codes")
    for code_df in pd.read_csv(os.path.join(data_folder, "CHARTEVENTS.csv.gz"), low_memory=False, chunksize=1000):
        code_rows = []
        for i,row in code_df.iterrows():
            code_rows.append([
                row.SUBJECT_ID,
            ])
        import pdb; pdb.set_trace()
        new_code_df = pd.DataFrame(code_rows, columns=code_columns)
        with open(os.path.join(new_data_folder, 'medical_codes.csv'), 'a') as f:
            new_code_df.to_csv(f, header=False, index=False)
        count += len(code_df)
        print(count)
"""

if __name__ == '__main__':
    if not os.path.exists(new_data_folder):
        os.mkdir(new_data_folder)
#    create_report_csv(data_folder, new_data_folder)
    create_code_csv(data_folder, new_data_folder)