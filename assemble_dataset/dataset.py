import pandas as pd
from tqdm import tqdm

class MedicalDataset:
    # saves to files in a folder called name
    @classmethod
    def create(cls, patient_ids, **kwargs):
        data = []
        iterator = tqdm(patient_ids, total=len(patient_ids))
        counts = cls.init_count_dict()
        for patient_id in iterator:
            datapoints = cls.get_datapoints(patient_id, counts, **kwargs)
            data.extend(datapoints)
            counts['num_datapoints'] += len(datapoints)
            iterator.set_postfix(counts)
        return pd.DataFrame(data, columns=cls.columns())

    @classmethod
    def init_count_dict(cls):
        return {'num_datapoints':0}

    @classmethod
    def columns(cls):
        raise NotImplementedError

    @classmethod
    def get_datapoints(cls, patient_id, counts, **kwargs):
        raise NotImplementedError

class Patient:
    def __init__(self, patient_id, reports=None, codes=None, admissions=None):
        self.reports = reports[reports.patient_id == patient_id].sort_values('date')\
            if reports is not None else None
        self.codes = codes[codes.patient_id == patient_id].sort_values('date')\
            if codes is not None else None
        self.admissions = admissions[admissions.patient_id == patient_id].sort_values('date_start')\
            if admissions is not None else None

    def compile_reports(self, after_date=None, before_date=None):
        if self.reports is None:
            raise Exception
        valid_rows = self.reports
        if after_date is not None:
            valid_rows = valid_rows[valid_rows.date >= after_date]
        if before_date is not None:
            valid_rows = valid_rows[valid_rows.date < before_date]
        return valid_rows
        """
        prev_date = None
        for i,row in valid_rows.iterrows():
            days = 0 if prev_date is None else (row.date-prev_date).days
            prev_date = row.date
            concatenated_reports += '\n\n'
            concatenated_reports += (' ' + time_separator)*days
            concatenated_reports += ' ' + report_separator
            concatenated_reports += ' ' + row.text
        return concatenated_reports
        """

    def targets(self, code_mapping=None):
        if self.codes is None:
            raise Exception
        # TODO: change this to using groupbys
        unique_codes = self.get_codes()
        targets = {}
        skipped = 0
        for code_type,code in unique_codes:
            code_str = str((code_type, code)).replace('.','')
            if code_mapping is not None:
                if code_str not in code_mapping.keys():
                    """
                    df = pd.DataFrame(list(code_mapping.keys()))
                    # create a list of similar codes
                    similar_codes = list(df[df[0].str.startswith("(\'%s\', \'%s" % (code_type, code)) & (df[0].str.len()==len(code_str)+1)][0])
                    code_length = len(eval(code_str)[1])
                    similar_codes += sum([list(df[df[0] == code_str[:l]][0]) for l in range(len(code_str)-3,len(code_str)-2-code_length,-1)], [])
                    if len(similar_codes) > 0:
                        # map to where a similar code maps to
                        mapped_code = code_mapping[similar_codes[0]]
                        if mapped_code is None:
                            continue
                        print("WARNING: unknown code being mapped: %s to %s" % (code_str, mapped_code)) # TODO: change this to an actual warning
                        print("    Similar Codes: "+str(similar_codes)) # TODO: change this to an actual warning
                    else:
                        # skip code
                        print("WARNING: code %s is unknown, skipping it!" % code_str) # TODO: change this to an actual warning
                        skipped += 1
                        continue
                    """
                    # skip code
                    skipped += 1
                    # print("WARNING: code %s is unknown, skipping it!" % code_str) # TODO: change this to an actual warning
                    continue
                else:
                    mapped_code = code_mapping[code_str]
                    if mapped_code is None:
                        # code mapping says to drop the code
                        continue
            else:
                mapped_code = code_str
            code_rows = self.codes[(self.codes.code_type == code_type) & (self.codes.code == code)]
            targets[mapped_code] = code_rows if mapped_code not in targets.keys() else\
                                   pd.concat([targets[mapped_code], code_rows], 0).sort_values('date')
        targets = [(target,rows) for target,rows in targets.items()]
        return targets, skipped

    def get_codes(self, after_date=None, before_date=None):
        codes = self.codes
        if after_date is not None:
            codes = codes[codes.date >= after_date]
        if before_date is not None:
            codes = codes[codes.date < before_date]
        return [(row.code_type,row.code) for i,row in codes[['code_type','code']].drop_duplicates().iterrows()]
