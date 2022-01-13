import pytest
import os

from click.testing import CliRunner
from src.models.predict_model import main as predict_model

import pandas as pd


def test_model_exist():
    with pytest.raises(NameError, match="No model found at.*"):
        runner = CliRunner()
        runner.invoke(
            predict_model,
            [
                "src/models/configs/no_exp_testing.yaml",
                "src/models/predictions/input.txt",
                "src/models/predictions/out.txt",
            ],
            catch_exceptions=False,
        )


@pytest.mark.skipif(not os.path.exists("models/exp1"), reason="Model files not found")
def test_input_extension():
    with pytest.raises(TypeError, match="Invalid input file type.*"):
        runner = CliRunner()
        runner.invoke(
            predict_model,
            [
                "src/models/configs/exp1.yaml",
                "src/models/predictions/input_bad_extension.text",
                "src/models/predictions/out.txt",
            ],
            catch_exceptions=False,
        )


@pytest.mark.skipif(not os.path.exists("models/exp1"), reason="Model files not found")
def test_output_extension():
    with pytest.warns(UserWarning, match="Invalid output file type.*"):
        runner = CliRunner()
        runner.invoke(
            predict_model,
            [
                "src/models/configs/exp1.yaml",
                "src/models/predictions/empty_input.txt",
                "src/models/predictions/out.text",
            ],
            catch_exceptions=False,
        )


@pytest.mark.skipif(not os.path.exists("models/exp1"), reason="Model files not found")
def test_create_output_txt():
    if os.path.isfile("src/models/predictions/out.txt"):
        os.remove("src/models/predictions/out.txt")

    runner = CliRunner()
    runner.invoke(
        predict_model,
        [
            "src/models/configs/exp1.yaml",
            "src/models/predictions/input.txt",
            "src/models/predictions/out.txt",
        ],
        catch_exceptions=False,
    )

    # Test output is created
    assert os.path.isfile("src/models/predictions/out.txt"), "Output not created"

    # Test output has as many lines as input
    with open("src/models/predictions/input.txt", "r") as in_f:
        in_num_lines = len(in_f.read().split("\n"))

    with open("src/models/predictions/out.txt", "r") as out_f:
        out_num_lines = len(out_f.read().split("\n"))

    assert (
        in_num_lines == out_num_lines
    ), "The size of the output differs from the input"


@pytest.mark.skipif(not os.path.exists("models/exp1"), reason="Model files not found")
def test_create_output_csv():
    if os.path.isfile("src/models/predictions/out.csv"):
        os.remove("src/models/predictions/out.csv")

    runner = CliRunner()
    runner.invoke(
        predict_model,
        [
            "src/models/configs/exp1.yaml",
            "src/models/predictions/input.txt",
            "src/models/predictions/out.csv",
        ],
        catch_exceptions=False,
    )

    # Test output is created
    assert os.path.isfile("src/models/predictions/out.txt"), "Output not created"

    # Test output has as many lines as input
    with open("src/models/predictions/input.txt", "r") as in_f:
        input_content = in_f.read()
        in_num_lines = len(input_content.split("\n"))

    out_df = pd.read_csv("src/models/predictions/out.csv")

    assert in_num_lines == len(out_df), "The size of the output differs from the input"

    # Test input row is equal to input
    out_text = "\n".join(out_df["description"].values)

    assert out_text == input_content, "Output has different content as input"
