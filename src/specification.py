"""End point construction interface for AI Writer APIS."""
from dataclasses import dataclass, field
from typing import Any, Mapping, Callable, Optional, Union, List

# Requested arguments for the model, and their types to cast

@dataclass
class EndPointArgs:
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    temperature: Optional[float] = None
    api_key: Optional[str] = None

    def as_dict(self) -> Mapping[str, Any]:
        """Returns the arguments as a dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class OpenAIEndPointArgs(EndPointArgs):
    model: str = "text-davinci-003"
    n:Optional[int] =  None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    best_of: Optional[int] = None
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
    del inputs
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