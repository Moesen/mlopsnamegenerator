import logging
import os

import click
import pandas as pd
import yaml
from architectures import SimpleGPT
from tqdm import tqdm
from transformers import GPT2Tokenizer


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def main(config_file: click.Path, input_file: click.Path, output_file: click.Path):
    """Does the prediction using a configuration from an experiment
    over the input_file and saves it in ouput_file

    Args:
        config_file (Path): path from where to load the configuraion
            !!must be .yaml!!
        input_file (Path): path from where to load the descriptions
            !!must be .txt or .csv!!
        config_file (Path): path to save the output in !!must be .txt
            or .csv!! if other format is inserted, the output will
            not be saved and will give a WARNING to the user
    """

    logger = logging.getLogger(__name__)
    logger.info("Reading config")

    # Only accepts yaml as config file
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_name = config["training"]["output_folder"]
    tokenizer_model = config["training"]["model"]
    separator = config["model"]["sep"]

    logger.info("Loading tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model")
    model = SimpleGPT.PokemonModel(
        transformers_model=model_name, eos_token_id=tokenizer.eos_token_id
    )

    # Accept either .txt with the description per line
    # or csv, with a column called 'descrption'
    _, input_file_extension = os.path.splitext(input_file)
    if input_file_extension == ".txt":
        with open(input_file, "r") as f:
            raw_input = f.read().split("\n")
    elif input_file_extension == ".csv":
        # Need to have 'description'
        raw_input = list(pd.read_csv(input_file)["description"])

    raw_input_sep = [x + separator for x in raw_input]

    decoded_out = []

    logger.info(f"Processing inputs from {input_file}")

    for input in tqdm(raw_input_sep, desc="Processing inputs"):

        encoded = tokenizer.encode(input, return_tensors="pt")
        output = model.model.generate(encoded, max_length=len(encoded[0]) + 10)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        decoded_out.append(decoded.split(separator)[-1])

    _, output_file_extension = os.path.splitext(output_file)

    logger.info(f"Saving output to {output_file}")

    # Warn users if not using one of the two allowed extensions
    if output_file_extension == ".txt":
        with open(output_file, "w") as f:
            f.write("\n".join(decoded_out))

    elif output_file_extension == ".csv":
        pd.DataFrame(
            zip(raw_input, decoded_out), columns=["description", "name"]
        ).to_csv(output_file, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
