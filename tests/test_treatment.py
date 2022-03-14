from src.data.make_naive_dataset import load_dataset
from src.data.treatment import Treatment
from src.data.make_naive_dataset import get_dataset_folder_names

_, dataset_name, raw_data_path, processed_data_path = get_dataset_folder_names()
dataframe = load_dataset(raw_data_path, processed_data_path, dataset_name).copy()
treatment = Treatment(dataframe, "stroke")


def test_columns_exist():
    assert treatment.column_exist('age') is True


def test_delete_column():
    df = dataframe.copy()
    treatment_test = Treatment(df, 'stroke')
    treatment_test.delete_column(['age'])
    if 'age' not in df.columns:
        test_check = 1
    else:
        test_check = 0
    assert test_check == 1
