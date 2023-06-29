from typing import Any, Optional, List
import tiktoken


class GPTTokenizer:

    def __init__(self,
                 model: str = "text-davinci-003",
                 max_tokens: int = 4096):
        self.model = model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.encoding_for_model("text-davinci-003")

        self.max_tokens = max_tokens

    def token_count(self, text: str, **kwargs) -> int:
        return len(self.tokenizer.encode(text, **kwargs))

    def encode(self, text: str, **kwargs) -> Any:
        text = self.truncate(text, **kwargs)
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, tokens: Any, **kwargs) -> str:
        return self.tokenizer.decode(tokens, **kwargs)

    def decode_batch(self, tokens: Any, **kwargs) -> List[str]:
        return self.tokenizer.decode_batch(tokens, **kwargs)

    def encode_batch(self, texts: List[str], **kwargs) -> Any:
        truncated = [self.truncate(text, **kwargs) for text in texts]
        return self.tokenizer.encode_batch(truncated, **kwargs)

    def truncate(self, text: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        if not max_tokens:
            max_tokens = self.max_tokens
            if not max_tokens:
                return text

        encoded = self.tokenizer.encode(text, **kwargs)[:max_tokens]

        return self.tokenizer.decode(encoded)
