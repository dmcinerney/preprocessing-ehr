# preprocessing-ehr

This repo is divided into 3 folders:

* `unify_format` - designed to unify the format of different ehr datasets using dataset specific scripts
* `icd_codes` - designed to do any further preprocessing of the icd_codes into structures used when constructing datasets
* `assemble_dataset` - designed to assemble the unified format data produced from the scripts in `unify_format` into various types of datasets


## Setup

Install the requirements:

    pip install -r requirements.txt

## Unify Format

### MIMIC-III

In order to preprocess ehr from MIMIC-III simply download the data and unzip it into a folder, and in `unify_format/convert_mimic.py` change the variable `data_folder` to point to the unziped folder, and change the variable `new_data_folder` to wherever you would like the semi-processed data to go.  Then execute the script:

    python unify_format/convert_mimic.py

## Assemble Dataset

### MIMIC-III

In order to assemble the preprocessed data into a dataset, execute the following command:

    python assemble_dataset/reports_and_codes_dataset.py <PATH>

where `<PATH>` is the path to the folder that the unify format script outputed to.
