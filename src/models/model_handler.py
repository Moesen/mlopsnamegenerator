from ts.torch_handler.base_handler import BaseHandler
from transformers import GPT2Tokenizer
import os
import pandas as pd
import torch

from src.models.architectures import SimpleGPT

class ModelHandler(BaseHandler):
    """
    Custom model handler implementation
    """


    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token   
        self.initialized = False    
        self.device = None
        self.model = None 


    def initialize(self, context):
        manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        self.model = SimpleGPT.PokemonModel(
            transformers_model=model_dir, eos_token_id=self.tokenizer.eos_token_id
        ).model

        self.model.to(self.device)
        self.model.eval()

        self.initialized = True

    def load_input(self, input_file, separator):
        """Accept either .txt with the description per line
        or csv, with a column called 'description',
        and returns the input

        Args:
            input_file (Path): Path to the input file
            separator (str): Separator to add to the input

        Returns:
            A list with each line of the input as element,
            with the separator and without it
        """

        preprocessed_data = input_file[0].get("data")
        if not preprocessed_data:
            preprocessed_data = input_file[0].get("body")
        
        preprocessed_data = preprocessed_data.decode().split("\n")

        return [x + separator for x in preprocessed_data]


    def handle(self, data, _):
        
        separator = " = /@\ = "
        raw_input_sep = self.load_input(data, separator)

        output = []

        for input_line in raw_input_sep:
            encoded = self.tokenizer.encode(input_line, return_tensors="pt")
            out = self.model.generate(encoded, max_length=len(encoded[0]) + 15)
            decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

            output.append(decoded.split(separator)[-1])
    
        return [output]