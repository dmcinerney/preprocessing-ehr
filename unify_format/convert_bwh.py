import pandas as pd
import os
import re
from datetime import datetime

tables = {
    'ContactInformation':'Con',
    'Demographics':'Dem',
    'Diagnosis':'Dia',
    'Discharge_Summaries':'Dis',
    'Encounters':'Enc',
    'Medication':'Med',
    'Mrn':'Mrn',
    'Operative_reports':'Opn',
    'Pathology':'Pat',
    'Phy':'Phy', # TODO: what is this?
    'Procedures':'Prc',
    'ProgressNote':'Prg',
    'Radiology':'Rad',
    'RadiologyTest':'Rdt',
    'ReasonForVisit':'Rfv',
    'VisitNote':'Vis'
}
folder = '/home/bwhbrain/Desktop/Text ML - McInerny'
filename = 'GY6_20190130_094741_%s.txt'

def preprocess_to_csv(folder, subfolder, filename, table, contains_report=False):
    with open(os.path.join(folder, filename % table), 'r') as infile:
        with open(os.path.join(folder, subfolder, filename % table), 'w') as outfile:
            for i,line in enumerate(infile):
                if (not contains_report) or i == 0:
                    line = line.replace('\n', '\r')
                else:
                    line = line.replace('[report_end]\n', '[report_end]\r')
                outfile.write(line)

report_files = set([
    'Discharge_Summaries',
    'Operative_reports',
    'Pathology',
    'ProgressNote',
    'Radiology',
    'VisitNote'
])
code_files = set([
    'Diagnosis'
    # TODO: should I add more?
])

def main():
    subfolder1 = 'PreprocessedTextFiles'
    subfolder2 = 'FinalPreprocessedData'
    os.mkdir(os.path.join(folder, subfolder1))
    for k,v in tables.items():
        preprocess_to_csv(folder, subfolder1, filename, v, contains_report=k in report_files)
    df = {k:pd.read_csv(os.path.join(folder, subfolder1, filename % v), delimiter='|', lineterminator='\r') for k,v in tables.items()}
    os.mkdir(os.path.join(folder, subfolder2))
    text_columns = ['patient_id', 'date', 'report_type', 'description', 'text', 'from_table', 'other_info']
    text_rows = []
    for k in report_files:
        for i,row in df[k].iterrows():
            results = re.findall('\d+', str(row.EMPI))
            if len(results) == 1:
                patient_id = int(results[0]) # this is the first one, so sometimes it has an extra newline at the beginning
            else:
                continue
            # date = row.Report_Date_Time # TODO: make this into a number for sorting purposes
            date = pd.to_datetime(datetime.strptime(row.Report_Date_Time, '%m/%d/%Y %I:%M:%S %p'))
            report_type = from_table = k
            description = row.Report_Description
            text = row.Report_Text
            other_info = {'report_status':row.Report_Status}
            text_rows.append([patient_id, date, report_type, description, text, report_type, from_table, other_info])
    pd.DataFrame(text_rows, columns=text_columns).to_csv(os.path.join(folder, subfolder2, 'medical_reports.csv'), index=False)
    code_columns = ['patient_id', 'date', 'flag', 'name', 'code_type', 'code', 'from_table', 'other_info']
    code_rows = []
    for k in code_files:
        for i,row in df[k].iterrows():
            patient_id = row.EMPI
            # date = row.Date # TODO: make this into a number for sorting purposes
            date = pd.to_datetime(datetime.strptime(row.Date, '%m/%d/%Y'))
            # need to split to handle flag, name, code_type, and code for different tables
            if k == 'Diagnosis':
                flag = row.Diagnosis_Flag
                name = row.Diagnosis_Name
                code_type = row.Code_Type
                code = row.Code
            else:
                raise NotImplementedError
            from_table = k
            code_rows.append([patient_id, date, flag, name, code_type, code, from_table, {}])
    pd.DataFrame(code_rows, columns=code_columns).to_csv(os.path.join(folder, subfolder2, 'medical_codes.csv'), index=False)
