# LangChain Chains - README

## 1. What are Chains?

In **LangChain**, a **Chain** is a pipeline that connects multiple components together (Prompts, Models, Parsers, Tools, etc.) so that data flows automatically from one step to another.

Instead of calling each component separately, chains let you **orchestrate the entire flow** in a clean and modular way.

---

## 2. Why Chains are Needed?

* **Automation**: Connects steps automatically.
* **Reusability**: Easy to reuse in different workflows.
* **Flexibility**: Build pipelines with sequence, parallel, or conditional flows.
* **Traceability**: Easier to debug and visualize with graphs.

Example Flow (Prompt → LLM → OutputParser → Final Result):

```
+-----------------+       +-----------+       +-----------------+
|   PromptTemplate|  -->  |  ChatOpenAI|  --> | StrOutputParser |
+-----------------+       +-----------+       +-----------------+
```

---

## 3. Runnables in Chains

Behind the scenes, LangChain uses the concept of **Runnables**:

* Every component (`PromptTemplate`, `ChatOpenAI`, `StrOutputParser`) is a **Runnable**.
* You can connect them using the `|` operator.
* This makes it easy to build pipelines without writing boilerplate code.

---

## 4. Pipeline Flow Types

LangChain supports different types of pipelines:

1. **Simple (Linear)**: Step 1 → Step 2 → Step 3.
2. **Sequential**: Multiple steps executed in order.
3. **Parallel**: Multiple steps executed simultaneously.
4. **Conditional**: Execution depends on conditions.

---

## 5. Types of Chains

### 5.1 Simple Chain

**Flow**: `Prompt → LLM → Parser`

Used for direct tasks such as generating a single output from an input. Example: generating facts about a topic.

Flowchart:

```
Input → Prompt → LLM → Parser → Output
```

---

### 5.2 Sequential Chain

**Flow**: `Topic → LLM (Detailed Report) → LLM (Summary in 5 Points)`

Sequential chains execute steps in order, where the output of one step becomes the input to the next.

Flowchart:

```
+-------+       +--------------------+       +---------------------+
| Topic |  -->  | Detailed Report LLM|  -->  | Summary (5 Points)  |
+-------+       +--------------------+       +---------------------+
```

Use case: Research → Report → Condensed Summary.

---

### 5.3 Parallel Chain

**Flow**: `Document → LLM → (Notes + Quiz in Parallel) → Merge into New LLM → Output`

Parallel chains allow multiple branches to run at the same time and then merge their outputs.

Flowchart:

```
                +-------- Notes Prompt --------+
               /                                \
Document --> LLM                                Merge --> Final Output
               \                                /
                +-------- Quiz Prompt  --------+
```

Use case: From a single document, generate **study notes** and **quiz questions** in parallel, then combine them into a unified study guide.

---

## 6. Key Takeaways

* A **Chain** is simply a pipeline of `Runnables`.
* The `|` operator connects components together.
* The chain handles data flow automatically.
* You can design **Sequential**, **Parallel**, and even **Conditional** pipelines.
* Flowcharts help visualize the execution clearly.
* `get_graph()` can generate ASCII representations of pipelines.

---

## 7. References

* [LangChain Docs - Chains](https://www.langchain.com/docs/expression_language/cookbook/chains)
* [LangChain Runnables](https://www.langchain.com/docs/expression_language/interface)
