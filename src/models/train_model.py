import logging
import os

import hydra
import wandb
from architectures import SimpleGPT
from datasets import Dataset, load_dataset
from hydra.utils import get_original_cwd
from transformers import GPT2Tokenizer, Trainer, TrainingArguments

import torch


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


@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg: dict):
    """Trains a given neural network with the parameters from cfg
    with the data from the data folder and saves the trained model
    in the output folder.

    Args:
        cfg (dict): configuration parameters for model and training
    Return:
        None
    """
    model_config = cfg.configs.model
    training_config = cfg.configs.training

    # Model parameters
    separator = model_config.sep
    offset = model_config.offset

    # Training parameters
    data_folder = os.path.join(get_original_cwd(), training_config.data_folder)
    output_folder = os.path.join(get_original_cwd(), training_config.output_folder)
    model_name = training_config.model
    batch_size = training_config.batch_size
    lr = training_config.lr
    epochs = training_config.epochs
    max_steps = training_config.max_steps
    seed = training_config.seed
    fp16 = training_config.fp16 and torch.cuda.is_available()

    wandb.init(project="test-project", entity="mlopsnamegenerator")

    logging.info("Loading tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    logging.info("Loading Model")
    model = SimpleGPT.PokemonModel(
        transformers_model=model_name, eos_token_id=tokenizer.eos_token_id
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

    max_length = longest_description + offset
    token_function = get_tokenize_function(tokenizer, separator, max_length)
    tokenized_dataset = dataset.map(token_function, batched=True)

    train_dataset = (
        tokenized_dataset["train"]
        .select(range(100))
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
        save_strategy="no",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        report_to="wandb",
        fp16=fp16,
        fp16_full_eval=fp16,
    )


    trainer = Trainer(
        model=model.model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logging.info("Training")
    trainer.train()

    logging.info("Finished training")

    logging.info("Training info")

    log_history = trainer.state.log_history

    for log_train, log_val in zip(log_history[::2], log_history[1::2]):
        logging.info(
            f"Epoch: {int(log_train['epoch'])}\t"
            + f"Training Loss: {log_train['loss']:.4f}\t"
            + f"Validation Loss:{log_val['eval_loss']:.4f}"
        )

    logging.info("Saving model")
    model.model.save_pretrained(output_folder)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
