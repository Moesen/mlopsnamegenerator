# -*- coding: utf-8 -*-
import logging
import os
import re

import click
import numpy as np
import pandas as pd


@click.command()
@click.argument("input_folder", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path())
@click.option("--seed", type=int, default=42, help="Seed for shuffling dataset")
def main(input_folder: click.Path, output_folder: click.Path, seed: int):
    """Loads the raw data from the input_folder and saves the processed
    dataset in output_folder.

    Args:
        input_folder (click.Path): path where the CSV with the raw data is
        output_folder (click.Path): path where the processed dataset is
            saved
        seed (int): Seed for the random shuffle of data
    Return:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info("Cleaning data descriptions")

    input_file_path = os.path.join(input_folder, "raw_descriptions.csv")
    df = pd.read_csv(input_file_path)
    df["description"] = df.apply(clean_descriptions, axis=1)
    df["name"] = df["name"].str.title()

    df = df.sample(frac=1, random_state=seed)

    train, test, val = np.split(df, [int(0.6 * len(df)), int(0.8 * len(df))])

    train_filepath = os.path.join(output_folder, "train.csv")
    test_filepath = os.path.join(output_folder, "test.csv")
    val_filepath = os.path.join(output_folder, "val.csv")

    train.to_csv(train_filepath, index=False)
    test.to_csv(test_filepath, index=False)
    val.to_csv(val_filepath, index=False)


def clean_descriptions(row: pd.Series):
    """Cleans the descriptions from the raw data.
    Replaces "\\n" with " " and the name of the Pokémon with "this Pokémon".

    Args:
        row (pd.Series): row of the CSV
    Return:
        cleaned (str): cleaned description
    """
    description = row["description"]
    cleaned = re.sub("\n", " ", description)

    name = row["name"]
    cleaned = re.sub(name, "this Pokémon", cleaned, flags=re.IGNORECASE)
    return cleaned


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
