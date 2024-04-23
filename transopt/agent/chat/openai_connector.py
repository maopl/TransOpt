import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import yaml
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from transopt.utils.log import logger


def get_prompt(file_name="prompt") -> str:
    """Reads a prompt from a file."""
    current_dir = Path(__file__).parent
    file_path = current_dir / file_name
    
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

class Message(BaseModel):
    """Model for LLM messages"""
    role: str  # The role of the message author (system, user, assistant, or function).
    content: Optional[Union[str, List[Dict]]] = None  # The message content.
    metrics: Dict[str, Any] = {}  # Metrics for the message.
    
    def get_content_string(self) -> str:
        """Returns the content as a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return json.dumps(self.content)
        return ""

    def to_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude_none=True, exclude={"metrics"})
        # Manually add the content field if it is None
        if self.content is None:
            _dict["content"] = None
        return _dict

    def log(self, level: Optional[str] = None):
        """Log the message to the console."""
        _logger = getattr(logger, level or "debug")
        _logger(f"============== {self.role} ==============")
        if self.content:
            _logger(self.content)


class OpenAIChat:
    # model: str = "gpt-4"
    model: str = "gpt-3.5-turbo"
    api_key: str = "sk-eGYDsI7kGLAVM9bs2585D337E51b48FbA30d88B0Fa8a1571"
    base_url: str = "https://aihubmix.com/v1"
    history: List[Message]

    def __init__(self, client_kwargs: Optional[Dict[str, Any]] = None):
        self.client_kwargs = client_kwargs or {}
        self.history = []
        
    @property
    def client(self):
        """Lazy initialization of the OpenAI client."""
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url, **self.client_kwargs)

    def invoke_model(self, messages: List[Message]) -> ChatCompletion:
        self.history.extend(messages)
        return self.client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in self.history],
        )

    def get_response(self, messages: List[Message]) -> str:
        logger.debug("---------- OpenAI Response Start ----------")
        # Log messages for debugging
        # for m in messages:
        #     m.log()

        response = self.invoke_model(messages=messages)

        # Parse response
        response_message = response.choices[0].message
        response_role = response_message.role
        response_content = response_message.content

        # Create assistant message
        assistant_message = Message(
            role=response_role or "assistant",
            content=response_content,
        )

        # Add assistant message to messages
        self.history.append(assistant_message)
        assistant_message.log()

        logger.debug("---------- OpenAI Response End ----------")
        # Return content if no function calls are present
        if assistant_message.content is not None:
            return assistant_message.get_content_string()
        return "Something went wrong, please try again."
