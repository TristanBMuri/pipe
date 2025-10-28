import os
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from langfuse import Langfuse, get_client
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langfuse._client.observe import observe
from pydantic import SecretStr

# A dummy class to represent the structure of query results from a vector database.
@dataclass
class QueryResult:
    meta_context: str
    context: str


def create_llm(config, max_tokens=None, timeout=None):
    provider = config.get("provider")
    api_key = None

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=config["model"],
            temperature=config["temperature"],
            max_output_tokens=max_tokens,
            timeout=timeout,
            max_retries=config["max_retries"],
            google_api_key=api_key
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=config["model"],
            temperature=config["temperature"],
            timeout=timeout,
            max_retries=config["max_retries"],
            api_key=SecretStr(api_key) if api_key else None
        )
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        return ChatDeepSeek(
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=config["max_retries"],
            api_key=SecretStr(api_key) if api_key else None
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class LangfusePrompter:
    """
    A class to manage fetching prompts from Langfuse and invoking LLMs with them.
    In v3, this class is simplified to focus on prompt and LLM logic,
    while tracing is handled by the calling context (e.g., in main.py).
    """
    def __init__(self, prompt_name: str, target_tag: str = "production", max_tokens: int = None, timeout: int = None):
        """
        Initializes the prompter by fetching the specified prompt from Langfuse
        and setting up the corresponding LLM.

        Args:
            prompt_name (str): The name of the prompt to fetch from Langfuse.
            target_tag (str, optional): The specific version/label of the prompt (e.g., "production"). Defaults to "production".
            max_tokens (int, optional): Max tokens for the LLM.
            timeout (int, optional): Timeout for the LLM call.
        """
        self.langfuse = get_client()
        self.prompt_name = prompt_name

        # Fetch the prompt from Langfuse, with a fallback to the latest version
        try:
            self.langfuse_prompt = self.langfuse.get_prompt(
                name=prompt_name,
                label=target_tag
            )
            logging.info(f"Successfully fetched prompt '{prompt_name}' with tag '{target_tag}'.")

        except Exception as e:
            logging.error(f"Could not fetch prompt '{prompt_name}' with tag '{target_tag}'. Falling back to latest. Error: {e}")
            self.langfuse_prompt = self.langfuse.get_prompt(prompt_name)

        # Create the LLM instance using the config from the fetched prompt
        self.llm = create_llm(self.langfuse_prompt.config, max_tokens=max_tokens, timeout=timeout)
        # Use below for passing structured input to model
        # self.llm = self.llm.with_structured_output()
        logging.info(f"LLM created for prompt '{prompt_name}': {self.get_model()}")

    def _get_messages(self, **kwargs):
        """
        Formats the prompt template with the given inputs.
        """
        # formatted_messages = self.langchain_prompt_template.format_messages(**kwargs)
        better_format = self.langfuse_prompt.compile(**kwargs)
        return better_format

    def report_prompt(self, user_input: str, **kwargs):
        """
        Prepares and invokes the LLM for generating a structured report.

        Args:
            user_input: The main input can be text, audio file path or audio file.
                If direct_audio=False: user_input should be a audio file path.
                If direct_audio=True: user_input can be either a path to an audio file or an audio file object.
            vectordb_context (QueryResult): Context from a vector database.
            structure (str, optional): A specific structure to be included in the prompt.
            direct_audio (bool, optional): Whether the input is an audio file or path instead of text.

        Returns:
            The response from the LLM.
        """
        processed_input = user_input

        # Now use the processed input (either original text or transcribed audio)
        messages = self._get_messages(
            user_input=processed_input,
            **kwargs
        )
        res = self.llm.invoke(messages).content
        # worked better with bad json responses from text prompts but doesn't work with chat prompts
        # res = self.llm.invoke([messages]).content
        return JsonOutputParser().parse(res)



    @observe(name="text_prompt", as_type="generation")
    def text_prompt(self, text_input: str, **kwargs):
        """
        Prepares and invokes the LLM for a simple tagging/analysis task.

        Args:
            text_input (str): The text to be analyzed.

        Returns:
            The response from the LLM.
        """
        messages = self._get_messages(
            transcription_text=text_input,
            **kwargs
        )
        return self.llm.invoke(messages)

    def get_model(self) -> str:
        """Returns the model name of the configured LLM."""
        # Langchain models have either a 'model' or 'model_name' attribute
        return getattr(self.llm, 'model', getattr(self.llm, 'model_name', 'unknown'))
