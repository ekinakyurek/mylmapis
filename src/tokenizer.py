from typing import Optional
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

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def truncate(self, text: str, max_tokens: Optional[int] = None) -> str:
        if not max_tokens:
            max_tokens  =self.max_tokens
        encoded = self.tokenizer.encode(text)[:max_tokens]
        return self.tokenizer.decode(encoded)
