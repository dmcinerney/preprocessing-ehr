class Patient:
    def __init__(self, patient_id, reports=None, codes=None):
        self.reports = reports[reports.patient_id == patient_id].sort_values('date')\
            if reports is not None else None
        self.codes = codes[codes.patient_id == patient_id].sort_values('date')\
            if codes is not None else None

    def concatenate_reports(self, before_date=None):
        if self.reports is None:
            raise Exception
        report_separator = '<report_separator>'
        time_separator = '<time_separator>'
        concatenated_reports = ''
        valid_rows = self.reports[self.reports.date < before_date]\
                     if before_date is not None else self.reports
        prev_date = None
        for i,row in valid_rows.iterrows():
            days = 0 if prev_date is None else (row.date-prev_date).days
            prev_date = row.date
            concatenated_reports += '\n\n'
            concatenated_reports += (' ' + time_separator)*days
            concatenated_reports += ' ' + report_separator
            concatenated_reports += ' ' + row.text
        return concatenated_reports

    def targets(self, code_mapping=None):
        if self.codes is None:
            raise Exception
        # TODO: change this to using groupbys
        unique_codes = self.codes[['code_type', 'code']].drop_duplicates()
        targets = {}
        for i,row in unique_codes.iterrows():
            code = str((row.code_type, row.code)).replace('.','')
            if code_mapping is not None and code not in code_mapping.keys():
                continue
            code_rows = self.codes[(self.codes.code_type == row.code_type) & (self.codes.code == row.code)]
            if code_mapping is not None:
                targets[code_mapping[code]] = code_rows if code_mapping[code] not in targets.keys() else\
                                              pd.concat([targets[code_mapping[code]], code_rows], 0)
            else:
                targets[code] = code_rows
        targets = [(target,rows) for target,rows in targets.items()]
        return targets
