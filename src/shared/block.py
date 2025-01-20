import hashlib

from .enums.common import AbstractEnum

class BlockType(AbstractEnum):
    MODEL = "model"
    GENESIS = "genesis"

class Block:
    def __init__(self) -> None:
        raise Exception("Block Class can't be concrete!")
    
    def calculate_hash(self):
        varList = [attr for attr in vars(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        attr =  [vars(self)[elem] for elem in varList]
        data_string = ""
        for attribute in attr:
            data_string += str(attribute)

        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def get_block_type(self):
        raise NotImplementedError("get_block_type")
    
    def to_json(self):
        raise NotImplementedError("to_json")
    
    def get_block_hash(self):
        raise NotImplementedError("get_block_hash")
    
    def get_previous_hash(self):
        raise NotImplementedError("get_previous_hash")
    
    @staticmethod
    def from_json(cls, json_string: str):
        raise NotImplementedError("from_json")