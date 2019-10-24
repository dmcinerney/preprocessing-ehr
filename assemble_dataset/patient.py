import pandas as pd

class Patient:
    def __init__(self, patient_id, reports=None, codes=None, admissions=None):
        self.reports = reports[reports.patient_id == patient_id].sort_values('date')\
            if reports is not None else None
        self.codes = codes[codes.patient_id == patient_id].sort_values('date')\
            if codes is not None else None
        self.admissions = admissions[admissions.patient_id == patient_id].sort_values('date_start')\
            if admissions is not None else None

    def concatenate_reports(self, after_date=None, before_date=None):
        if self.reports is None:
            raise Exception
        report_separator = '<report_separator>'
        time_separator = '<time_separator>'
        concatenated_reports = ''
        valid_rows = self.reports
        if after_date is not None:
            valid_rows = valid_rows[valid_rows.date >= after_date]
        if before_date is not None:
            valid_rows = valid_rows[valid_rows.date < before_date]
        prev_date = None
        for i,row in valid_rows.iterrows():
            days = 0 if prev_date is None else (row.date-prev_date).days
            prev_date = row.date
            concatenated_reports += '\n\n'
            concatenated_reports += (' ' + time_separator)*days
            concatenated_reports += ' ' + report_separator
            concatenated_reports += ' ' + row.text
        return concatenated_reports

    def targets(self, code_mapping=None, code_graph=None):
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
                    # create a list of similar codes that have the original code as a prefix but have an additional character
                    similar_codes = list(df[df[0].str.startswith("(\'%s\', \'%s" % (code_type, code)) & (df[0].str.len()==len(code_str)+1)][0])
                    if len(similar_codes) > 0:
                        # map to where a similar code maps to
                        mapped_code = code_mapping[similar_codes[0]]
                        if code_graph is not None:
                            # if there is a graph take the predecessor of the mapped code bc it should be on the level of generality that the original code had
                            predecessors = list(code_graph.predecessors(mapped_code))
                            if len(predecessors) > 0:
                                mapped_code = predecessors[0]
                        print("WARNING: unknown code being mapped: %s to %s" % (code_str, mapped_code)) # TODO: change this to an actual warning
                        print("    Similar Codes: "+str(similar_codes)) # TODO: change this to an actual warning
                    else:
                        # skip code
                        print("WARNING: code %s is unknown, skipping it!" % code_str) # TODO: change this to an actual warning
                        print("    Similar Codes: "+str(similar_codes)) # TODO: change this to an actual warning
                        continue
                    """
                    # skip code
                    skipped += 1
                    print("WARNING: code %s is unknown, skipping it!" % code_str) # TODO: change this to an actual warning
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
