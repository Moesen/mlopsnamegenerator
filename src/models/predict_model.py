import logging
import os

from architectures import SimpleGPT
from transformers import GPT2Tokenizer


def main():

    print(os.getcwd())

    model_name = "models/exp1"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = SimpleGPT.PokemonModel(
        transformers_model=model_name, eos_token_id=tokenizer.eos_token_id
    )

    separator = r" = /@\ = "

    with open("src/models/predictions/input.txt", "r") as f:
        raw_input = f.read().split("\n")

    raw_input = [x + separator for x in raw_input]

    for input in raw_input:

        encoded = tokenizer.encode(input, return_tensors="pt")

        output = model.model.generate(encoded, max_length=len(encoded[0]) + 10)

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        print(decoded.split(separator)[-1])


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
