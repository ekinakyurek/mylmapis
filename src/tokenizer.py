from typing import Any, Optional, List
import tiktoken


class GPTTokenizer:

    def __init__(self,
                 model: str = "text-davinci-003",
                 max_tokens: int = 4096,
                 new_special_tokens: Optional[List[str]] = None,):
        self.model = model
        try:
            tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            tokenizer = tiktoken.encoding_for_model("text-davinci-003")

        if new_special_tokens:
            max_tokens_value = tokenizer.max_token_value
            extended_tokenizer = tiktoken.Encoding(
                # If you're changing the set of special tokens, make sure to use a different name
                # It should be clear from the name what behaviour to expect.
                name=model + "-special",
                pat_str=tokenizer._pat_str,
                mergeable_ranks=tokenizer._mergeable_ranks,
                special_tokens={
                    **tokenizer._special_tokens,
                    "<|startoftext|>": max_tokens_value + 1,
                    "<|vid_start|>": max_tokens_value + 2,
                    "<|vid_end|>": max_tokens_value + 3,
                }
            )
            tokenizer = extended_tokenizer

        self.tokenizer = tokenizer
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
