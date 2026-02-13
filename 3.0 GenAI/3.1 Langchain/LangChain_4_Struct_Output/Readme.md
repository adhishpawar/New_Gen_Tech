
# 📘 Structured Output in LangChain

### Generative AI using LangChain (`with_structured_output`)

Large Language Models (LLMs) usually return **unstructured text**, which is hard to parse reliably in production systems.  
LangChain solves this by allowing **structured outputs**, where the LLM is forced to return data in a predefined format.

This is achieved using:

```python
structured_model = model.with_structured_output(schema)
```

LangChain supports **three main schema types**:

- `TypedDict`
    
- `Pydantic`
    
- `JSON Schema`
    

---

## 🔹 Why Structured Output?

In GenAI applications, structured outputs are critical for:

- Reliable downstream processing
    
- Agent workflows (tool calling)
    
- API responses
    
- Analytics & dashboards
    
- Reducing hallucinations
    

---

## 🔹 1. TypedDict

`TypedDict` comes from Python’s `typing` module and provides **type hints only**.

### ✅ What it does

- Defines the **shape of the output**
    
- Helps with IDE autocomplete and readability
    

### ❌ What it does NOT do

- No runtime validation
    
- No default values
    
- No type conversion
    

### Example

```python
from typing import TypedDict

class SentimentOutput(TypedDict):
    sentiment: str
    confidence: float
```

### ✅ Use TypedDict if:

- You only need **basic structure enforcement**
    
- You **trust the LLM** to return correct data
    
- You want **zero overhead**
    
- You are prototyping quickly
    

---

## 🔹 2. Pydantic

Pydantic is a **data validation and parsing library** widely used in production Python systems.

### ✅ What it does

- Strong runtime validation
    
- Default values
    
- Enum constraints
    
- Automatic type conversion
    
- Python object output
    

### Example

```python
from pydantic import BaseModel
from typing import Literal

class SentimentOutput(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float = 0.0
```

### ✅ Use Pydantic if:

- You need **strict validation**
    
- LLM output must follow **business rules**
    
- You want **safe production systems**
    
- You want automatic conversions (e.g., `"100"` → `100`)
    

---

## 🔹 3. JSON Schema

JSON Schema is a **language-agnostic standard** for defining JSON structure.

### ✅ What it does

- Defines structure + validation
    
- Works across languages
    
- No Python dependency required
    

### ❌ What it does NOT do

- No Python objects
    
- No automatic defaults unless explicitly defined
    

### Example

```python
sentiment_schema = {
  "type": "object",
  "properties": {
    "sentiment": {
      "type": "string",
      "enum": ["positive", "neutral", "negative"]
    },
    "confidence": {
      "type": "number"
    }
  },
  "required": ["sentiment"]
}
```

### ✅ Use JSON Schema if:

- You need **cross-language compatibility**
    
- You don’t want extra Python libraries
    
- You are exposing outputs via APIs
    
- You are integrating with frontend or non-Python systems
    

---

## 🔹 Comparison Table

|Feature|TypedDict|Pydantic|JSON Schema|
|---|---|---|---|
|Basic structure|✅|✅|✅|
|Runtime validation|❌|✅|✅|
|Default values|❌|✅|❌|
|Type conversion|❌|✅|❌|
|Python objects|❌|✅|❌|
|Cross-language compatibility|❌|❌|✅|
|Production safety|❌|✅|✅|

---

## 🔹 When to Use What?

### ✅ Use **TypedDict** when:

- You only need structure
    
- Fast prototyping
    
- Minimal overhead
    
- You trust the LLM output
    

### ✅ Use **Pydantic** when:

- Validation is critical
    
- You need defaults
    
- Output feeds business logic
    
- Production-grade GenAI systems
    

### ✅ Use **JSON Schema** when:

- Cross-language usage is required
    
- Frontend + backend integration
    
- You want a standard schema format
    
- No Python-specific dependency
    

---

## 🔹 `with_structured_output()` – Method Parameter

LangChain internally supports **different enforcement strategies** depending on the LLM.

### 1️⃣ JSON Mode (Schema-based Output)

Used by models like **Claude / Gemini**

```python
model.with_structured_output(schema, method="json_mode")
```

- Forces the model to return **pure JSON**
    
- Best for:
    
    - APIs
        
    - Analytics
        
    - Validation-first workflows
        
- No tool execution
    

---

### 2️⃣ Function Calling (Tool Calling)

Used mainly by **OpenAI models**

```python
model.with_structured_output(schema, method="function_calling")
```

- Model selects and calls a function
    
- Essential for **Agents**
    
- Enables:
    
    - Calculator
        
    - Search tools
        
    - Database queries
        
    - External APIs
        

📌 Example use case:

- AI agent deciding when to call a calculator instead of answering in text.
    

---

## 🔹 Important Note on LLMs & Structured Output

LLMs **do not inherently understand structure**.  
They generate text.

👉 `with_structured_output()` **constrains the output**, but:

- It **does NOT guarantee logic correctness**
    
- Validation still matters
    
- Pydantic / JSON Schema add safety layers
    

---

## 🔹 Summary

- Structured output is **mandatory** for production GenAI
    
- Choose schema based on **validation needs & system design**
    
- JSON Schema is the **only cross-language option**
    
- Function calling is required for **agent-based systems**
    
- JSON mode is best for **pure data extraction**
    
