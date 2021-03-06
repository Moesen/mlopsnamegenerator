from transformers import GPT2LMHeadModel


class PokemonModel:
    """Class for inputting a pretrained network from the
    transformers package, and choosing
    a model to finetune, based on a train dataset
    """

    def __init__(self, transformers_model: str, eos_token_id):
        """Inits a small class saving settings for
        the model

        Args:
            transformers_model (str): Type of
            tokenizer (tokenizer): The token of the tokenizer
        """
        self.model = GPT2LMHeadModel.from_pretrained(
            transformers_model, pad_token_id=eos_token_id
        )