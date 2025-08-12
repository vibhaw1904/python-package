# LangChain Nugen.in Integration

[![PyPI version](https://badge.fury.io/py/langchain-nugen.svg)](https://badge.fury.io/py/langchain-nugen)
[![Python Support](https://img.shields.io/pypi/pyversions/langchain-nugen.svg)](https://pypi.org/project/langchain-nugen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official LangChain integration for [Nugen.in](https://nugen.in) LLMs, providing seamless access to Nugen.in's language models through LangChain's standardized interface.

## 🚀 Features

- **Full LangChain Compatibility**: Works with all LangChain components (chains, agents, memory, etc.)
- **Streaming Support**: Real-time token generation with `stream()` method
- **Chat Interface**: Proper message handling with `ChatNugen` class
- **Environment Variables**: Secure credential management
- **Type Safety**: Full type hints and Pydantic validation
- **Async Support**: Coming soon in v0.2.0

## 📦 Installation

```bash
pip install langchain-nugen
```

## 🔧 Quick Start

### Basic Usage

```python
from langchain_nugen import ChatNugen
from langchain_core.messages import HumanMessage

# Initialize with API key
chat = ChatNugen(
    api_key="your-nugen-api-key",
    model_name="nugen-flash-instruct"
)

# Send a message
messages = [HumanMessage(content="Hello, how are you?")]
response = chat(messages)
print(response.content)
```

### Using Environment Variables (Recommended)

```bash
export NUGEN_API_KEY="your-nugen-api-key"
```

```python
from langchain_nugen import ChatNugen

# Automatically uses NUGEN_API_KEY from environment
chat = ChatNugen(model_name="nugen-flash-instruct")
response = chat([HumanMessage(content="Hello!")])
```

### Streaming Responses

```python
from langchain_core.messages import HumanMessage

chat = ChatNugen()
messages = [HumanMessage(content="Tell me a story about AI")]

for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```

## 🔗 LangChain Integration Examples

### With LLMChain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_nugen import ChatNugen

template = """You are a helpful assistant. Answer the following question:

Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
chat = ChatNugen()
chain = LLMChain(llm=chat, prompt=prompt)

response = chain.run(question="What is machine learning?")
print(response)
```

### With ConversationChain

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_nugen import ChatNugen

chat = ChatNugen()
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=chat, memory=memory)

# Have a conversation
response1 = conversation.predict(input="Hi, my name is Alice")
response2 = conversation.predict(input="What's my name?")
```

### With Agents

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain_nugen import ChatNugen

# Initialize tools
search = DuckDuckGoSearchRun()
tools = [search]

# Create agent
chat = ChatNugen()
agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use agent
response = agent.run("What's the weather like in New York today?")
```

## ⚙️ Configuration

### ChatNugen Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | None | Nugen.in API key (required) |
| `model_name` | `str` | `"nugen-flash-instruct"` | Model identifier |
| `base_url` | `str` | `"https://api.dev-nugen.in"` | API base URL |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0-2.0) |
| `max_tokens` | `int` | `1000` | Maximum tokens to generate |
| `top_p` | `float` | `1.0` | Top-p sampling parameter |

### Example with Custom Parameters

```python
chat = ChatNugen(
    api_key="your-api-key",
    model_name="nugen-ultra-instruct",
    temperature=0.9,
    max_tokens=2000,
    top_p=0.95
)
```

## 🧪 Advanced Usage

### Batch Processing

```python
from langchain_core.messages import HumanMessage

chat = ChatNugen()
messages_list = [
    [HumanMessage(content="What is AI?")],
    [HumanMessage(content="What is ML?")],
    [HumanMessage(content="What is DL?")]
]

responses = chat.batch(messages_list)
for response in responses:
    print(response.content)
```

### With Custom System Messages

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a expert Python developer."),
    HumanMessage(content="How do I implement a binary search?")
]

chat = ChatNugen()
response = chat(messages)
print(response.content)
```

### Error Handling

```python
from langchain_nugen import ChatNugen
from langchain_core.messages import HumanMessage

try:
    chat = ChatNugen(api_key="invalid-key")
    response = chat([HumanMessage(content="Hello")])
except Exception as e:
    print(f"Error: {e}")
```

## 🔒 Security Best Practices

1. **Use Environment Variables**: Store your API key in environment variables, not in code
2. **Rotate Keys Regularly**: Update your API keys periodically
3. **Limit Permissions**: Use API keys with minimal required permissions
4. **Monitor Usage**: Keep track of your API usage and costs

```python
# ✅ Good - using environment variables
import os
os.environ["NUGEN_API_KEY"] = "your-api-key"
chat = ChatNugen()

# ❌ Bad - hardcoding API key
chat = ChatNugen(api_key="your-api-key-here")
```

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/unit_tests/

# Run integration tests (requires NUGEN_API_KEY)
export NUGEN_API_KEY="your-api-key"
pytest tests/integration_tests/

# Run all tests
pytest
```

## 📚 API Reference

### ChatNugen Class

#### Methods

- `__call__(messages: List[BaseMessage]) -> BaseMessage`: Generate response
- `stream(messages: List[BaseMessage]) -> Iterator[BaseMessage]`: Stream response
- `batch(messages_list: List[List[BaseMessage]]) -> List[BaseMessage]`: Batch process

#### Properties

- `_llm_type: str`: Returns `"nugen-chat"`
- `_identifying_params: Dict[str, Any]`: Model configuration parameters

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/yourusername/langchain-nugen.git
cd langchain-nugen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black langchain_nugen/
isort langchain_nugen/

# Type checking
mypy langchain_nugen/

# Run linters
flake8 langchain_nugen/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

- **Documentation**: [GitHub README](https://github.com/yourusername/langchain-nugen)
- **Issues**: [GitHub Issues](https://github.com/yourusername/langchain-nugen/issues)
- **Nugen.in API Docs**: [https://docs.nugen.in/](https://docs.nugen.in/)
- **LangChain Docs**: [https://python.langchain.com/docs/](https://python.langchain.com/docs/)

## 📈 Changelog

### v0.1.0 (2024-01-15)
- Initial release
- ChatNugen implementation with BaseChatModel
- Streaming support
- Environment variable configuration
- Comprehensive test suite
- Full LangChain ecosystem compatibility

---

**Made with ❤️ for the LangChain community**