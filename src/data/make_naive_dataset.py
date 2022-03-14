# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from src.data.treatment import Treatment
import logging
from config.definitions import ROOT_DIR

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
log = logging.getLogger("__name__")


def load_dataset(raw, processed, data_name, raw_data_name=None):
    """
    Load processed data if file doesn't exist else Copy a .csv
    file from raw data folder to processed data folder and load it.
    params:
    - raw_data_path: string, path to raw data
    - processed_data_path: string, path to processed data folder
    - data_filename: string, data name
    - verbose: bool,  to print dataset information
    return:
    - data: .csv data processed
    """
    if len(os.listdir(processed)) == 0:  # check if processed data file is empty
        shutil.copyfile(os.path.join(raw, raw_data_name), os.path.join(processed, data_name))
        log.info("Warning: no preprocess data, we are copying raw data.")
    data_loaded = pd.read_csv(os.path.join(processed, data_name))
    log.info("Data '{}' loaded with data shape: {}".format(data_name, data_loaded.shape))
    return data_loaded


def get_dataset_folder_names():
    raw = 'healthcare-dataset-stroke-data.csv'
    data_name = 'naive_healthcare-dataset-stroke-data.csv'
    raw_path = os.path.join(ROOT_DIR, 'data', 'raw')
    processed_path = os.path.join(ROOT_DIR, 'data', 'processed')

    return raw, data_name, raw_path, processed_path


def main(input_filepath, output_filepath, data_name, raw_data_name=None):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    Preprocessing data raw to make naive cleaned dataset.
    """
    dataframe = load_dataset(input_filepath, output_filepath, data_name, raw_data_name=raw_data_name)  # load raw datas
    treatment = Treatment(dataframe, 'stroke')  # create treatment class
    treatment.to_categorical()  # replace categorical columns
    treatment.missing_values('bmi')  # replace 'bmi' missing value
    if treatment.column_exist('id'):
        treatment.delete_column(['id'])  # remove 'id' feature
    dataframe.to_csv(output_filepath + data_name, index=False)  # save modified .csv file
    log.info('preprocessing data set from raw data saved in processed data.\n Data shape: {}'.format(dataframe.shape))


if __name__ == '__main__':
    raw_dataset_name, dataset_name, raw_data_path, processed_data_path = get_dataset_folder_names()
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(raw_data_path, processed_data_path, dataset_name, raw_data_name=raw_dataset_name)
