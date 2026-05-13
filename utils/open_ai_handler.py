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
    reasoning: Optional[str] = None,  # e.g. "low", "medium", "high"
    temperature: float = 0.7,
    max_output_tokens: int = 500,
    system_prompt: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Sends a prompt to OpenAI and returns the text output.

    Parameters
    ----------
    prompt : str
        User prompt
    model : str
        Model name (e.g. gpt-4.1, gpt-4.1-mini, o4-mini)
    reasoning : str | None
        Reasoning level for reasoning-capable models ("low", "medium", "high")
    temperature : float
        Controls randomness (0.0 = deterministic, 1.0 = creative)
    max_output_tokens : int
        Max tokens in the response
    system_prompt : str | None
        Optional system instruction
    response_format : dict | None
        For structured output (e.g. {"type": "json_object"})

    Returns
    -------
    str
        Model output text
    """

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
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    # Enable reasoning if supported
    if reasoning is not None:
        response_kwargs["reasoning"] = {"effort": reasoning}

    # Structured outputs (JSON, etc.)
    if response_format is not None:
        response_kwargs["response_format"] = response_format

    response = client.responses.create(**response_kwargs)

    return response.output_text
