import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Union

import click
import pandas as pd
import requests
from tqdm import tqdm

# Poke API domain
POKE_API_DOMAIN = "https://pokeapi.co/api/v2/"
# Query to find whole pokedex
POKE_API_QUERY = "pokedex/national"
# Query to find select pokemon based on entry number
POKE_API_SELECT_QUERY = "pokemon-species/"


def download_species(queries: Tuple[int, str]) -> Union[list, dict]:
    data = []
    entry_number, name = queries
    query = f"{POKE_API_DOMAIN}{POKE_API_SELECT_QUERY}{entry_number}"

    response = requests.get(query)
    content = response.json()["flavor_text_entries"]

    for description in content:
        language = description["language"]["name"]
        text = description["flavor_text"]

        if language == "en":
            data.append([entry_number, name, text])
    return data


@click.command()
@click.argument("output_folder", type=click.Path())
@click.option(
    "--max_workers", type=int, default=32, help="Number of threads when downloading"
)
def main(output_folder: click.Path, max_workers: int):
    """Downloads the text descriptions from the PokeAPI and saves it in the
    output_folder.

    Args:
        output_folder (click.Path): path where the CSV with the
            data is saved
    Return:
        None
    """
    logger = logging.getLogger(__name__)
    file_path = os.path.join(output_folder, "raw_descriptions.csv")

    if not os.path.isfile(file_path):
        logger.info("Downloading data")
        response = requests.get(POKE_API_DOMAIN + POKE_API_QUERY)
        entries = response.json()["pokemon_entries"]

        queries = [
            (specie["entry_number"], specie["pokemon_species"]["name"])
            for specie in entries
        ]

        data = []
        with tqdm(total=len(entries)) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(download_species, query) for query in queries]
                for future in as_completed(futures):
                    pbar.update(1)
                    data.extend(future.result())

        df = pd.DataFrame(data, columns=["entry_name", "name", "description"])
        df.to_csv(file_path, index=False)

    else:
        logger.warning("Data already downloaded")
        warnings.warn(UserWarning("Data already downloaded"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
