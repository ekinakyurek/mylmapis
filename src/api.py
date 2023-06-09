import asyncio
import logging
import os
import pickle
from typing import List, Mapping, Any, Callable, Optional, Tuple
import openai
from .specification import EndPointSpec
from .utils import batch
from .utils import SafeFormat
from .tokenizer import GPTTokenizer
from .network_manager import async_retry_with_exp_backoff

JSON = Mapping[str, Any]
STRINGMAP = Mapping[str, str]

acreate = async_retry_with_exp_backoff(openai.Completion.acreate)

achatcreat = async_retry_with_exp_backoff(openai.ChatCompletion.acreate)

async def query_with_retry(inputs: List[str], **kwargs) -> List[JSON]:
    """Queries GPT API up to max_retry time to get the responses."""
    is_chat_model = "turbo" in kwargs.get("model", "") or "gpt-4" in kwargs.get("model", "")

    if is_chat_model:
        caller = achatcreat
        if len(inputs) != 1:
            tasks = [query_with_retry([prompt], **kwargs) for prompt in inputs]
            responses = await asyncio.gather(*tasks)
            return [response[0] for response in responses]
        else:
            kwargs["messages"] = [{"role": "user", "content": inputs[0]}]
            if "best_of" in kwargs:
                kwargs.pop("best_of")
    else:
        caller = acreate
        kwargs["prompt"] = inputs

    try:
        response = await caller(**kwargs)
    except Exception as msg:
        logging.warning("API Error: %s", str(msg))
        return [{"status": "ERROR"}] * len(inputs)

    choices = response["choices"]

    if is_chat_model:
        choices = [{"text": choice["message"]["content"]} for choice in choices]

    return choices

def freeze(o):
    if isinstance(o, dict):
        return frozenset({k: freeze(v) for k, v in o.items()}.items())
    if isinstance(o, list):
        return frozenset([freeze(v) for v in o])
    return o


def cacher(cache_path: str, func):
    """Caches the results of the GPT."""
    async def wrapper(inputs: List[str], **kwargs):
        kwargs_key = kwargs.copy()
        del kwargs_key['api_key']
        del kwargs_key['logit_bias']
        kwargs_key['stop'] = tuple(kwargs_key['stop'])

        kwargs_key = frozenset(kwargs_key.items())

        # find unseen inputs
        unseen = []

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        else:
            cache = {}
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)

        for inp in inputs:
            key = (inp, kwargs_key)
            if key not in cache:
                unseen.append(inp)

        # get results for unseen inputs
        outputs = await func(unseen, **kwargs)

        # reload cache
        with open(cache_path, "rb") as handle:
            cache = pickle.load(handle)

        # add results to cache
        for inp, out in zip(unseen, outputs):
            key = (inp, kwargs_key)
            cache[key] = out

        # save cache
        with open(cache_path, "wb") as handle:
            pickle.dump(cache, handle)

        # get results from cache
        outputs = []
        for inp in inputs:
            key = (inp, kwargs_key)
            outputs.append(cache[key])

        return outputs
    return wrapper



def create_prompt(template: str, tokenizer: GPTTokenizer, inp: STRINGMAP) -> str:
    truncated_input = {}
    count = tokenizer.token_count(template)
    # print("Template token count: ", count)
    for k, text in inp.items():
        if not isinstance(text, str):
            text = ""
        # print(f"{k} count: ", tokenizer.token_count(text))
        new_count = count + tokenizer.token_count(text)
        if new_count > tokenizer.max_tokens:
            # logging.warning("Truncating input.")
            truncated_input[k] = tokenizer.truncate(text, tokenizer.max_tokens - count)
            count += tokenizer.token_count(truncated_input[k])
        else:
            truncated_input[k] = text
            count = new_count

    prompt = template.format_map(SafeFormat(truncated_input))

    return prompt


def create_api_function(spec: EndPointSpec, cache_path: Optional[str] = None) -> Callable:
    """High level function"""

    spec_kwargs = spec.args.end_point_args.as_dict()

    tokenizer_args = {
        "model": spec_kwargs.get("model", None),
        "max_tokens": spec.args.max_input_tokens,
    }

    tokenizer = GPTTokenizer(**tokenizer_args)

    if cache_path:
        caller = cacher(cache_path, query_with_retry)
    else:
        caller = query_with_retry

    async def api_function(
        inputs: List[JSON],
        **kwargs,
    ) -> List[JSON]:
        """API Function"""
        # Preprocess and translate text to English
        api_kwargs = spec_kwargs.copy()

        if kwargs:
            for k, value in kwargs.items():
                api_kwargs[k] = value

        processed = spec.preprocesser(inputs)

        # print("inputs: ", inp)
        prompts = [create_prompt(spec.template, tokenizer, inp) for inp in processed]
        # print("prompts: ", prompts)
        # Call model API with prompts
        generations = await caller(prompts, **api_kwargs)

        outputs = spec.postprocesser(inputs, generations)

        return outputs

    return api_function


async def run(
    inputs: List[JSON],
    api_function: Callable[[List[JSON]], List[JSON]],
    api_keys: Optional[Tuple[str]] = None,
    num_workers: int = 0,
    batch_size: int = 20,
) -> List[JSON]:
    """Runs the API function in parallel on the inputs."""
    if not api_keys:
        api_keys = [openai.api_key]

    if num_workers == 0:
        num_workers = len(api_keys)

    outputs = []

    for task_inputs in batch(batch(inputs, batch_size), num_workers):
        tasks = [
            api_function(list(inp), api_key=api_keys[i % len(api_keys)])
            for i, inp in enumerate(task_inputs)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)

        for result in results:
            outputs += result

    return outputs


async def run_pipeline(
    data: List[JSON],
    pipeline: List[EndPointSpec],
    cache_path: Optional[str] = None,
    **kwargs,
):
    """Runs the pipeline on the data."""
    inputs = data
    for _, spec in enumerate(pipeline):
        api_function = create_api_function(spec, cache_path=cache_path)
        # print("before api call", inputs)
        inputs = await run(inputs, api_function, **kwargs)
        # print("after api call", inputs)

    return inputs


if __name__ == "__main__":
    def test():
        """Test script."""
        import os
        from .specification import APIArgs, OpenAIEndPointArgs

        api_keys = os.getenv("OPENAI_API_KEY_POOL").split(",")

        def postprocesser(inps, outs):
            del inps
            return [{"text": out["text"]} for out in outs]

        pipeline = [
            EndPointSpec(
                name="test",
                template="{text}",
                args=APIArgs(
                    OpenAIEndPointArgs(model="gpt-3.5-turbo", max_tokens=16, n=1, temperature=1.0),
                    max_input_tokens=1024,
                    truncate_input=True,
                ),
                postprocesser=postprocesser,
            ),
            # EndPointSpec(
            #     name="test",
            #     template="{text}",
            #     args=APIArgs(
            #         OpenAIEndPointArgs(model="gpt-3.5-turbo", max_tokens=16, n=1, temperature=1.0),
            #         max_input_tokens=1024,
            #         truncate_input=True,
            #     ),
            # ),
        ]

        data = [{"text": "Can you count from 1 to 10?"}] * 1

        task = run_pipeline(data, pipeline, api_keys=api_keys)

        result = asyncio.run(task)

        print(result)
    test()