import json
from mma.agent import Agent
from mma.utils import parse_json

class ResourceMemoryAgent(Agent):
    def __init__(
        self,
        **kwargs
    ):
        # load parent class init 
        super().__init__(**kwargs)
