import pickle
import click
from transformers import GPT2Tokenizer
import yaml
import logging

from src.models.architectures import SimpleGPT


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.argument("out_model_file", type=click.Path())
@click.argument("out_token_file", type=click.Path())
def main(config_file, out_model_file, out_token_file):

    logger = logging.getLogger(__name__)
    logger.info("Reading config")

    # Only accepts yaml as config file
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_name = config["training"]["output_folder"]
    tokenizer_model = config["training"]["model"]

    logger.info("Loading tokenizer")

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Dumping tokenizer")

    pickle.dump(tokenizer, open(out_token_file, "wb"))

    logger.info("Loading model")

    model = SimpleGPT.PokemonModel(
        transformers_model=model_name, eos_token_id=tokenizer.eos_token_id
    ).model

    logger.info("Dumping model")

    pickle.dump(model, open(out_model_file, "wb"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()