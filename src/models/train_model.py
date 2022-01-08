import logging
import os
from pathlib import Path
from typing import Callable, List

import click
import pandas as pd
from architectures import SimpleGPT
from datasets import Dataset, load_dataset
from dotenv import find_dotenv, load_dotenv
from transformers import GPT2Tokenizer, Trainer, TrainingArguments


def get_tokenize_function(tokenizer: GPT2Tokenizer, separator: str, max_length: int):
    """Function used when mapping dataset with a certain tokenization.

    Args:
        tokenizer (GPT2Tokenizer): The object used to tokenize
        separator (str): Special characters added so the model can
            learn that what comes after is the name/answer
    Returns:
        (Callable): Tokenizer function used in dataset.map
    """

    def tokenize_function(text: Dataset):
        """Function to tokenize an input in format
        [description] + [separator] + [name].

        Args:
            text (Dataset): Dataset with 'name' and 'description'
                columns
        """
        output = [
            pkmn_desc + separator + pkmn_name
            for pkmn_name, pkmn_desc in zip(text["name"], text["description"])
        ]

        results = tokenizer(output, max_length=max_length, padding="max_length")
        results["labels"] = results["input_ids"].copy()
        return results

    return tokenize_function


@click.command()
@click.argument("data_folder", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path())
@click.option(
    "--model", type=str, help="Model from the transformer package", default="gpt2"
)
@click.option("--sep", type=str, default=" = /@\ = ", help="Seperator if line too long")
@click.option(
    "--offset", type=int, default=15, help="Offset of accepted max length of sentence"
)
def main(
    data_folder: str,
    output_folder: str,
    model: str,
    sep: str,
    offset: int,
):
    """Trains a given neural network on a given train file and
    returns the pretrained model in a folder.

    Args:
        data_folder (str): Path to the data folder.
            Must have 2.csv files called train.csv and val.csv
        output_folder (str): Path to where the model is saved.
            Please choose a not yet generated folder, as filenames are
            non-configurable and will override previous models!
        model (str): Name of model, go in and choose from available models on hugging face
        sep (str): Separator
        offset (int): The max additional space, a sentence can have,
            than the longest sentence in given dataset
    Return:
        None
    """

    batch_size = 16
    lr = 5e-05
    epochs = 3
    max_steps = -1
    seed = 42

    logging.info("Loading tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    separator = sep

    logging.info("Loading Model")
    model = SimpleGPT.PokemonModel(
        transformers_model=model, eos_token_id=tokenizer.eos_token_id
    )

    logging.info("Loading Dataset")
    train_file = os.path.join(data_folder, "train.csv")
    val_file = os.path.join(data_folder, "val.csv")

    dataset = load_dataset(
        "csv",
        data_files={"train": train_file, "validation": val_file},
    )

    all_descriptions = (
        dataset["train"]["description"] + dataset["validation"]["description"]
    )
    longest_description = max(len(x.split(" ")) for x in all_descriptions)
    print(longest_description)

    max_length = longest_description + offset
    print(max_length)
    token_function = get_tokenize_function(tokenizer, separator, max_length)
    tokenized_dataset = dataset.map(token_function, batched=True)

    train_dataset = (
        tokenized_dataset["train"]
        .select(range(10))
        .remove_columns(["name", "description", "entry_name"])
    )

    val_dataset = (
        tokenized_dataset["validation"]
        .select(range(10))
        .remove_columns(["name", "description", "entry_name"])
    )

    logging.info("Loading Trainer")
    train_args = TrainingArguments(
        output_dir=output_folder,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=max_steps,
        seed=seed,
        save_strategy="epoch",
        evaluation_strategy="epoch",
    )

    # To see what is going on
    compute_metrics = lambda eval_pred: print(eval_pred)

    trainer = Trainer(
        model=model.model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logging.info("Training")
    trainer.train()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()
