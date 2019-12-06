from unify_format.convert_mimic_cxr import main as unify_mimic_cxr
from assemble_dataset.reports_and_codes_dataset import main as make_reports_codes
from assemble_dataset.reports_and_codes_aligned_dataset import main as make_reports_codes_aligned
from assemble_dataset.reports_to_seq_dataset import main as make_reports_seq
from icd_codes.process_codes import main as process_codes
from icd_codes.radiology_codes import main as radiology_codes

if __name__ == '__main__':
    make_reports_codes_aligned()
#    make_reports_codes()
#    radiology_codes()
#    process_codes()
#    make_reports_seq()
#    unify_mimic_cxr()
