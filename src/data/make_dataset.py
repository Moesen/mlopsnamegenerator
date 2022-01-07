# -*- coding: utf-8 -*-
import logging
import re

import click
import numpy as np
import pandas as pd


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: click.Path, output_filepath: click.Path):
    """
    Loads the raw data from the input_filepath and saves the processed
    dataset in output_filepath.

    Args:
        input_filepath (click.Path): path where the CSV with the raw data is
        output_filepath (click.Path): path where the processed dataset is
        saved
    Return:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info("Cleaning data descriptions")

    df = pd.read_csv(f"{input_filepath}/raw_descriptions.csv")
    df["description"] = df.apply(clean_descriptions, axis=1)
    df["name"] = df["name"].str.title()

    df = df.sample(frac=1, random_state=42)

    train, test, val = np.split(df, [int(0.6 * len(df)), int(0.8 * len(df))])

    train.to_csv(f"{output_filepath}/train.csv", index=False)
    test.to_csv(f"{output_filepath}/test.csv", index=False)
    val.to_csv(f"{output_filepath}/val.csv", index=False)


def clean_descriptions(row: pd.Series):
    """
    Cleans the descriptions from the raw data.
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
