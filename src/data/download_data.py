import logging
import os

import click
import pandas as pd
import requests
from tqdm import tqdm


@click.command()
@click.argument("output_folder", type=click.Path())
def main(output_folder: click.Path):
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
        data = []
        domain = "https://pokeapi.co/api/v2/"
        query = "pokedex/national"

        response = requests.get(domain + query)
        entries = response.json()["pokemon_entries"]

        for specie in tqdm(entries, desc="Downloading data"):
            entry_number = specie["entry_number"]
            name = specie["pokemon_species"]["name"]

            query = f"pokemon-species/{entry_number}"

            response = requests.get(domain + query)
            content = response.json()["flavor_text_entries"]

            for description in content:
                language = description["language"]["name"]
                text = description["flavor_text"]

                if language == "en":
                    data.append([entry_number, name, text])

        df = pd.DataFrame(data, columns=["entry_name", "name", "description"])
        df.to_csv(file_path, index=False)

    else:
        logger.info("Data already downloaded")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
