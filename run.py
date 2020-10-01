from unify_format.convert_bwh import main as unify_bwh
from unify_format.convert_mimic import main as unify_mimic
from unify_format.convert_mimic_cxr import main as unify_mimic_cxr
from assemble_dataset.reports_and_codes_dataset import main as make_reports_codes
from assemble_dataset.reports_and_entities_dataset import main as make_reports_entities
from assemble_dataset.reports_and_codes_aligned_dataset import main as make_reports_codes_aligned
from assemble_dataset.reports_to_seq_dataset import main as make_reports_seq
from assemble_dataset.reports_txt_dataset import main as make_reports_txt
from icd_codes.process_codes import main as process_codes
from icd_codes.radiology_codes import main as radiology_codes

if __name__ == '__main__':
#    make_reports_txt()
#    make_reports_codes_aligned()
#    unify_bwh()
    unify_mimic()
#    make_reports_codes()
#    make_reports_entities()
#    radiology_codes()
#    process_codes()
#    make_reports_seq()
#    unify_mimic_cxr()
