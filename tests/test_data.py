import pytest
import os
import pandas as pd
from click.testing import CliRunner

from src.data.download_data import main as download_data
from src.data.make_dataset import main as make_dataset
from src.data.make_dataset import clean_descriptions



@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_files_created_raw():
    if os.path.isfile("data/raw/raw_descriptions.csv"):
        with pytest.warns(UserWarning):
            runner = CliRunner()
            runner.invoke(download_data, ["data/raw"])
    else:
        runner = CliRunner()
        runner.invoke(download_data, ["data/raw"])
        assert os.path.isfile("data/raw/raw_descriptions.csv"), "No data found"



@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_number_columns_raw():
    assert set(pd.read_csv("data/raw/raw_descriptions.csv").columns) == {"entry_name", "name", "description"}, "Not correct columns in raw data"



@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_no_missing_value_raw():
    assert not pd.read_csv("data/raw/raw_descriptions.csv").isnull().values.any(), "The raw data has missing values"



@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_data_cleaning():
    assert pd.read_csv("data/raw/raw_descriptions.csv").apply(clean_descriptions, axis=1).str.contains("\n").sum() == 0, "The cleaned data has line skips ('\n')"

@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_files_Created_processed():
    runner = CliRunner()
    runner.invoke(make_dataset, ["data/raw", "data/processed"])
    assert os.path.isfile("data/processed/train.csv") and os.path.isfile("data/processed/test.csv") and os.path.isfile("data/processed/val.csv"), "No data processed found"


@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_number_columns_train():
    assert set(pd.read_csv("data/processed/train.csv").columns) == {"entry_name", "name", "description"}, "Not correct columns in train data"



@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_number_columns_test():
    assert set(pd.read_csv("data/processed/test.csv").columns) == {"entry_name", "name", "description"}, "Not correct columns in test data"



@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_number_columns_val():
    assert set(pd.read_csv("data/processed/val.csv").columns) == {"entry_name", "name", "description"}, "Not correct columns in validation data"



@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_no_missing_value_train():
    assert not pd.read_csv("data/processed/train.csv").isnull().values.any(), "The train data has missing values"



@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_no_missing_value_test():
    assert not pd.read_csv("data/processed/test.csv").isnull().values.any(), "The test data has missing values"



@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_no_missing_value_val():
    assert not pd.read_csv("data/processed/val.csv").isnull().values.any(), "The validation data has missing values"



@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_data_size():
    raw_size = len(pd.read_csv("data/raw/raw_descriptions.csv"))
    train_size = len(pd.read_csv("data/processed/train.csv"))
    test_size = len(pd.read_csv("data/processed/test.csv"))
    val_size = len(pd.read_csv("data/processed/val.csv"))

    assert raw_size == train_size + test_size + val_size, "The processed data size is not the same as the raw data"

''' 
IDEAS FOR MORE TESTS THAT WE FOUND THAT MAY NOT BE NECESSARY
 - Test that the shuffle is the same using the same seed.
    Generating twice the data result in the same datasets
'''