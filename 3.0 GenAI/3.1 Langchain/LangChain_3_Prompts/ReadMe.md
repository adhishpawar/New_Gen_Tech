# LangChain Chat Prompt Templates - Detailed Notes

## 1. Introduction to LangChain Chat Prompts
LangChain is a framework that allows developers to easily build applications using large language models. One of its core features is managing chat prompts, which are structured conversations between the user (Human) and the AI.

A **Chat Prompt Template** helps in designing these interactions efficiently.

---

## 2. Key Components

### 2.1 Messages and Message Placeholders
- **Message**: Represents a single message in a conversation.
- **Message Placeholder**: Used to dynamically insert content (like user input) into the prompt.

Types of messages:
1. **HumanMessage**: Message from the user.
2. **AIMessage**: Message from the AI.
3. **SystemMessage**: System instructions that guide the AI behavior.

---

## 3. Single Message vs List of Messages

### 3.1 Single Message
- Represents only one message.
- Can be **static** or **dynamic**.

**Static Example:**
```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

message = HumanMessage(content="Hello! How are you?")
```

**Dynamic Example:**
```python
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("Hello, my name is {name}.")
])
formatted_prompt = chat_prompt.format_prompt(name="Adhish")
print(formatted_prompt.to_messages())
```

### 3.2 List of Messages
- Represents multiple messages in a conversation.
- Can combine **static** and **dynamic** messages.

**Static Messages Example:**
```python
from langchain.schema import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]
```

**Dynamic Messages Example:**
```python
from langchain.prompts.chat import (
    ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
)

chat_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("Hello {user_name}, how can I help you today?"),
    AIMessagePromptTemplate.from_template("Sure {user_name}, I can help with that.")
])
formatted_prompt = chat_prompt.format_prompt(user_name="Adhish")
messages = formatted_prompt.to_messages()
for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

---

## 4. Static vs Dynamic Prompts

### 4.1 Static Prompt
- Predefined text.
- Does not change at runtime.
- Used for fixed conversations.

Example:
```python
HumanMessage(content="Tell me a joke.")
AIMessage(content="Why did the chicken cross the road? To get to the other side!")
```

### 4.2 Dynamic Prompt
- Uses templates and variables.
- Can adapt based on user input or context.
- Uses `ChatPromptTemplate` and placeholders.

Example:
```python
chat_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("What is the capital of {country}?")
])
formatted_prompt = chat_prompt.format_prompt(country="India")
messages = formatted_prompt.to_messages()
```

---

## 5. Summary Table
| Type | Class | Description |
|------|-------|-------------|
| Single Static | HumanMessage / AIMessage | One fixed message |
| Single Dynamic | HumanMessagePromptTemplate / AIMessagePromptTemplate | Template with placeholders for one message |
| List Static | SystemMessage / HumanMessage / AIMessage | Predefined conversation sequence |
| List Dynamic | ChatPromptTemplate with multiple MessagePromptTemplates | Template with multiple messages and placeholders |

---

## 6. References
1. [LangChain Chat Prompt Documentation](https://www.langchain.com/docs/modules/prompts/chat)
2. [LangChain GitHub Repository](https://github.com/hwchase17/langchain)

---

*This document provides detailed notes on managing single and list-based chat prompts in LangChain, highlighting static and dynamic usage.*

