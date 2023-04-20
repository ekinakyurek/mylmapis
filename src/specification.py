"""End point construction interface for AI Writer APIS."""
from dataclasses import dataclass, field
from typing import Any, Mapping, Callable, Optional, Union, List

# Requested arguments for the model, and their types to cast

@dataclass
class EndPointArgs:
    max_tokens: int = 16
    n: int = 1
    temperature: float = 0.0
    api_key: Optional[str] = None


@dataclass
class OpenAIEndPointArgs(EndPointArgs):
    model: str = "text-davinci-003"
    n: int = 1
    top_p: float = 1.0
    temperature: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 16
    best_of: int = 1
    logit_bias: Mapping[str, float] = field(default_factory=dict)
    api_key: Optional[str] = None
    stop: Optional[Union[List[str], str]] = None


@dataclass
class APIArgs:
    end_point_args: EndPointArgs
    truncate_input: bool = True
    max_input_tokens: int = 4096



JSON = Mapping[str, Any]
STRINGMAP = Mapping[str, str]

def preprocesser(inputs: List[JSON]) -> List[STRINGMAP]:
    """Preprocesses the inputs."""
    return inputs

def postprocesser(inputs: List[JSON], outputs: List[JSON]) -> List[JSON]:
    """Postprocesses the inputs."""
    return outputs

@dataclass
class EndPointSpec:
    """End point specification structure."""
    name: str
    template: str = None
    template_file: Optional[str] = None
    args: APIArgs = field(default_factory=APIArgs)
    preprocesser: Callable[[List[JSON]], List[STRINGMAP]] = preprocesser
    postprocesser: Callable[[List[JSON]], List[JSON]] = postprocesser

    def __post_init__(self):
        if self.template_file:
            with open(self.template_file, "r", encoding="utf-8") as f:
                self.template = f.read()