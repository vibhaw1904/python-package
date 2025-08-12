"""Nugen.in Chat Model Implementation."""

from typing import Any, Dict, Iterator, List, Optional
import requests
import json
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, SecretStr, model_validator
from langchain_core.utils import get_from_dict_or_env


class ChatNugen(BaseChatModel):
    """Nugen.in Chat Model Integration.
    
    Example:
        .. code-block:: python
        
            from langchain_nugen import ChatNugen
            from langchain_core.messages import HumanMessage
            
            chat = ChatNugen(
                api_key="your-api-key",
                model_name="nugen-flash-instruct"
            )
            
            messages = [HumanMessage(content="Hello, how are you?")]
            response = chat(messages)
            print(response.content)
    """
    
    # Required fields
    api_key: SecretStr = Field(description="Nugen.in API Key")
    model_name: str = Field(default="nugen-flash-instruct", description="Model name")
    base_url: str = Field(default="https://api.nugen.in", description="API base URL")
    
    # Optional parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for randomness")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    
    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": True
    }
        
    @model_validator(mode='before')
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate environment variables and configuration."""
        api_key = get_from_dict_or_env(
            values, "api_key", "NUGEN_API_KEY"
        )
        values["api_key"] = SecretStr(api_key) if isinstance(api_key, str) else api_key
        return values
    
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "nugen-chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
    
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to Nugen.in format."""
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            elif isinstance(message, ChatMessage):
                prompt_parts.append(f"{message.role}: {message.content}")
            else:
                prompt_parts.append(f"Unknown: {message.content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response."""
        prompt = self._convert_messages_to_prompt(messages)
        
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v3/inference/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract response text
            if "choices" in result and len(result["choices"]) > 0:
                text = result["choices"][0].get("text", "")
            elif "response" in result:
                text = result["response"]
            else:
                text = str(result)
            
            # Create AIMessage response
            message = AIMessage(content=text)
            generation = ChatGeneration(message=message)
            
            # Add usage metadata if available
            usage = result.get("usage", {})
            llm_output = {"token_usage": usage, "model_name": self.model_name}
            
            return ChatResult(generations=[generation], llm_output=llm_output)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Nugen API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response."""
        prompt = self._convert_messages_to_prompt(messages)
        
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v3/inference/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                text = chunk_data["choices"][0].get("text", "")
                                if text:
                                    chunk = ChatGenerationChunk(
                                        message=AIMessageChunk(content=text)
                                    )
                                    if run_manager:
                                        run_manager.on_llm_new_token(text, chunk=chunk)
                                    yield chunk
                        except json.JSONDecodeError:
                            continue
                            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Nugen API streaming error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected streaming error: {str(e)}")