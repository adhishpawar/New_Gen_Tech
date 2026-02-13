# LangChain Models Notes

## 1. Difference Between LLM and ChatModel

| Feature              | LLM (OpenAI class)                                | ChatModel (ChatOpenAI class)                          |
|----------------------|---------------------------------------------------|------------------------------------------------------|
| **Usage**            | For traditional completion models (single prompt → single text output). | For modern chat-based models (messages → conversational response). |
| **Example Models**   | `text-davinci-003`, `gpt-3.5-turbo-instruct`      | `gpt-3.5-turbo`, `gpt-4`, `gpt-4o`, `gpt-4o-mini`    |
| **Input Format**     | Plain string prompt                               | List of messages (`SystemMessage`, `HumanMessage`)   |
| **Output Format**    | String                                            | ChatMessage object with `.content`                   |
| **LangChain Class**  | `langchain_openai.OpenAI`                         | `langchain_openai.ChatOpenAI`                        |

---

## 2. Packages and Base Classes

- **`langchain_openai.OpenAI`**
  - Inherits from `BaseLLM`
  - Designed for **completion** style models

- **`langchain_openai.ChatOpenAI`**
  - Inherits from `BaseChatModel`
  - Designed for **chat-based** models

- **Core Base Classes in LangChain**
  - `BaseLLM` → Abstract class for LLMs
  - `BaseChatModel` → Abstract class for chat models
  - `LLMResult` → Standardized output object for results

---

## 3. Important Parameters (Attributes)

### 🔹 Temperature
- Controls **randomness** of output.
- Range: `0.0` (deterministic, predictable) → `1.0+` (more random, creative).
- Example:
  ```python
  ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
  ```

### 🔹 max_completion_tokens
- Sets the **maximum number of tokens** the model is allowed to generate.
- Prevents overly long responses and controls cost.
- Example:
  ```python
  ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=256)
  ```

---

## 4. Example Usage

### ✅ Using LLM (completion model)
```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.3, max_tokens=200)
response = llm.invoke("Write a short note on Machine Learning.")
print(response)
```

### ✅ Using ChatModel (chat-based model)
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_completion_tokens=200)
response = chat.invoke([HumanMessage(content="What is the capital of India?")])
print(response.content)
```

---

## 5. Key Takeaways
- Use **`OpenAI` (LLM)** for single prompt → text completions.  
- Use **`ChatOpenAI` (ChatModel)** for structured chat-based conversations.  
- Always tune `temperature` and `max_completion_tokens` based on **creativity vs. control** needs.
