import logging
import os
from pathlib import Path
from typing import Callable, List

import click
import pandas as pd
from architectures import SimpleGPT
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from transformers import GPT2Tokenizer, Trainer, TrainingArguments


def get_tokenize_function(tokenizer: GPT2Tokenizer, separator: str, max_length: int):
    """Function used when mapping dataset with a certian tokenizastion

    Args:
        tokenizer (GPT2Tokenizer): The object used to tokenize
         separator (str): [description]
    Returns:
        (Callable): Tokenizer function used in dataset.map
    """

    def tokenize_function(text):
        output = [separator + pkmn_name for pkmn_name in text["name"]]
        results = tokenizer(
            text["description"], output, max_length=max_length, padding="max_length"
        )
        results["labels"] = results["input_ids"].copy()
        return results

    return tokenize_function


@click.command()
@click.argument("train_filepath", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path())
@click.option(
    "--model", type=str, help="Model from the transformer package", default="gpt2"
)
@click.option("--sep", type=str, default=" = /@\ = ", help="Seperator if line too long")
@click.option("--seed", type=int, default=42, help="Seed for shuffling dataset")
@click.option("--size", type=int, default=500, help="Size of training dataset")
@click.option(
    "--offset", type=int, default=10, help="Offset of accepted max length of sentence"
)
def main(
    train_filepath: str,
    output_folder: str,
    model: str,
    sep: str,
    seed: int,
    size: int,
    offset: int,
):
    """Trains a given neural network on a given train file and returns the pretrained
    model in a folder.

    Args:
        train_filepath (str): Path to the training file. (Has to be .csv!)
        output_folder (str): Path to where the model is saved.
            Please choose a not yet generated folder, as filenames are
            non-configurable and will override previous models!
        model (str): Name of model, go in and choose from available models on hugging face
        sep (str): Separator
        seed (int): Random seed when creating sub-dataset
        size (int): Size of training dataset
        offset (int): The max additional space, a sentence can have,
            than the longest sentence in given dataset
    """
    logging.info("Loading tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    separator = sep

    logging.info("Loading Model")
    model = SimpleGPT.PokemonModel(
        transformers_model=model, eos_token_id=tokenizer.eos_token_id
    )

    logging.info("Loading Dataset")
    dataset = load_dataset("csv", data_files=train_filepath)
    longest_description = max(
        len(x.split(" ")) for x in dataset["train"]["description"]
    )
    max_length = longest_description + offset
    token_function = get_tokenize_function(tokenizer, separator, max_length)
    tokenized_dataset = dataset.map(token_function, batched=True)

    train_dataset = (
        tokenized_dataset["train"]
        .shuffle(seed=seed)
        .select(range(size))
        .remove_columns(["name", "description", "entry_name"])
    )

    logging.info("Loading Trainer")
    train_args = TrainingArguments("test_trainer", label_names=None)

    # To see what is going on
    compute_metrics = lambda eval_pred: print(eval_pred)

    trainer = Trainer(model=model.model, args=train_args, train_dataset=train_dataset)

    logging.info("Training")
    trainer.train()

    logging.info(f"Saving model in {output_folder}")
    model.model.save_pretrained(output_folder)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()
