from openai import OpenAI
from typing import Optional, Dict, Any




def load_openai_client(api_key_file: str = "API.txt") -> OpenAI:
    """
    Loads the OpenAI API key from a file and returns a client instance.
    """
    with open(api_key_file, "r") as f:
        api_key = f.read().strip()

    if not api_key:
        raise ValueError("API key file is empty")

    return OpenAI(api_key=api_key)


def run_openai(
    prompt: str,
    model: str = "gpt-4.1",
    reasoning: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: int = 500,
    system_prompt: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:

    client = load_openai_client()

    input_content = []

    if system_prompt:
        input_content.append({
            "role": "system",
            "content": system_prompt
        })

    input_content.append({
        "role": "user",
        "content": prompt
    })

    response_kwargs = {
        "model": model,
        "input": input_content,
        "max_output_tokens": max_output_tokens,
    }

    # Only send temperature if model supports it
    if temperature is not None and not model.startswith("gpt-5"):
        response_kwargs["temperature"] = temperature

    # Only send reasoning if model supports it
    if reasoning is not None and not model.startswith("gpt-5"):
        response_kwargs["reasoning"] = {"effort": reasoning}

    if response_format is not None:
        response_kwargs["response_format"] = response_format

    response = client.responses.create(**response_kwargs)

    return response.output_text
