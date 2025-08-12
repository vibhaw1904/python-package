"""Unit tests for ChatNugen."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_nugen.chat_models.nugen import ChatNugen


def test_nugen_initialization():
    """Test ChatNugen initialization."""
    chat = ChatNugen(api_key="test-key", model_name="test-model")
    assert chat._llm_type == "nugen-chat"
    assert chat.model_name == "test-model"
    assert chat.temperature == 0.7
    assert chat.max_tokens == 1000


def test_identifying_params():
    """Test identifying parameters."""
    chat = ChatNugen(api_key="test-key", model_name="test-model", temperature=0.5)
    params = chat._identifying_params
    
    expected_params = {
        "model_name": "test-model",
        "temperature": 0.5,
        "max_tokens": 1000,
        "top_p": 1.0,
    }
    assert params == expected_params


def test_convert_messages_to_prompt():
    """Test message conversion to prompt format."""
    chat = ChatNugen(api_key="test-key")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        HumanMessage(content="How are you?")
    ]
    
    prompt = chat._convert_messages_to_prompt(messages)
    expected = (
        "System: You are a helpful assistant.\n"
        "Human: Hello\n"
        "Assistant: Hi there!\n"
        "Human: How are you?\n"
        "Assistant:"
    )
    assert prompt == expected


@patch('requests.post')
def test_nugen_generate_success(mock_post):
    """Test successful ChatNugen generation."""
    # Mock API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"text": " Hello! How can I help you today?"}],
        "usage": {"total_tokens": 15, "prompt_tokens": 5, "completion_tokens": 10}
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    # Test generation
    chat = ChatNugen(api_key="test-key")
    messages = [HumanMessage(content="Hello")]
    result = chat._generate(messages)
    
    # Verify results
    assert len(result.generations) == 1
    assert isinstance(result.generations[0].message, AIMessage)
    assert "Hello!" in result.generations[0].message.content
    assert result.llm_output["token_usage"]["total_tokens"] == 15
    
    # Verify API call
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["json"]["model"] == "nugen-flash-instruct"
    assert "Human: Hello\nAssistant:" in call_args[1]["json"]["prompt"]


@patch('requests.post')
def test_nugen_generate_with_stop_tokens(mock_post):
    """Test ChatNugen generation with stop tokens."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"text": "Hello there"}],
        "usage": {}
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    chat = ChatNugen(api_key="test-key")
    messages = [HumanMessage(content="Hello")]
    result = chat._generate(messages, stop=["!", "?"])
    
    # Verify stop tokens were passed
    call_args = mock_post.call_args
    assert call_args[1]["json"]["stop"] == ["!", "?"]


@patch('requests.post')
def test_nugen_generate_api_error(mock_post):
    """Test ChatNugen generation with API error."""
    # Mock API error
    mock_post.side_effect = Exception("API connection failed")
    
    chat = ChatNugen(api_key="test-key")
    messages = [HumanMessage(content="Hello")]
    
    with pytest.raises(Exception) as exc_info:
        chat._generate(messages)
    
    assert "Unexpected error: API connection failed" in str(exc_info.value)


def test_nugen_custom_parameters():
    """Test ChatNugen with custom parameters."""
    chat = ChatNugen(
        api_key="test-key",
        model_name="custom-model",
        temperature=0.9,
        max_tokens=500,
        top_p=0.8,
        base_url="https://custom.api.url"
    )
    
    assert chat.model_name == "custom-model"
    assert chat.temperature == 0.9
    assert chat.max_tokens == 500
    assert chat.top_p == 0.8
    assert chat.base_url == "https://custom.api.url"


def test_nugen_environment_variable():
    """Test ChatNugen initialization from environment variable."""
    with patch.dict('os.environ', {'NUGEN_API_KEY': 'env-test-key'}):
        chat = ChatNugen()
        assert chat.api_key.get_secret_value() == 'env-test-key'


def test_invalid_temperature():
    """Test ChatNugen with invalid temperature."""
    with pytest.raises(ValueError):
        ChatNugen(api_key="test-key", temperature=3.0)  # Above valid range


def test_invalid_max_tokens():
    """Test ChatNugen with invalid max_tokens."""
    with pytest.raises(ValueError):
        ChatNugen(api_key="test-key", max_tokens=0)  # Below valid range