Summarisation Strategy
# Conversation Summarisation Strategy

A lightweight Python module that automatically summarizes long conversations to keep them within an LLM's token limit while preserving important context.

## Overview

Large Language Models (LLMs) have a limited context window. As conversations grow, sending the entire chat history becomes inefficient and may exceed the model's token limit.

This project solves that problem by:

- Tracking conversation messages
- Estimating token usage
- Automatically compressing older messages
- Preserving important facts and topics
- Keeping recent conversation unchanged

The result is a conversation history that stays small while retaining the most useful information.

---

## Project Structure

```
.
├── conversation_manager.py        # Main conversation manager
├── test_conversation_manager.py   # Unit tests
├── test_run_output.txt            # Example test output
```

---

## Features

- Automatic token estimation
- Configurable token limit
- Automatic conversation compression
- Rule-based summarization
- Optional Claude AI summarizer support
- Conversation history management
- Compression history tracking
- Easy integration into chatbots and AI assistants

---

## How It Works

```
User Message
      │
      ▼
Add to Conversation
      │
      ▼
Calculate Total Tokens
      │
      ▼
Token Limit Reached?
      │
 ┌────┴────┐
 │         │
 No       Yes
 │         │
 │         ▼
 │   Select Older Messages
 │         │
 │         ▼
 │   Generate Summary
 │         │
 │         ▼
 │ Replace Old Messages
 │         │
 └────────► Keep Recent Messages
                 │
                 ▼
        Send Context to LLM
```

---

## Components

### 1. TokenCounter

Estimates the number of tokens used by text.

Functions:

- `count(text)`
- `count_messages(messages)`

---

### 2. Message

Represents one conversation message.

Fields:

- role
- content
- timestamp
- is_summary

---

### 3. ConversationManager

Responsible for:

- storing conversation
- monitoring token usage
- generating summaries
- returning optimized context

Main methods:

```python
add_message(role, content)

get_context()

total_tokens()

raw_total_tokens()

describe()
```

---

### 4. Default Summarizer

A lightweight summarizer that:

- extracts important facts
- detects preferences
- keeps names
- keeps budgets
- keeps deadlines
- keeps important topics

No external API is required.

---

### 5. Claude Summarizer (Optional)

The module also supports using Anthropic Claude for higher-quality summaries.

Simply provide a Claude client and use:

```python
make_claude_summarizer(client)
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd conversation-summarisation
```

No additional dependencies are required for the default summarizer.

For Claude summarization:

```bash
pip install anthropic
```

---

## Example Usage

```python
from conversation_manager import ConversationManager

manager = ConversationManager(token_limit=500)

manager.add_message("user", "My name is Alice.")
manager.add_message("assistant", "Nice to meet you!")

context = manager.get_context()

print(context)
```

---

## Automatic Compression

When the conversation exceeds the configured token limit:

1. Older messages are selected.
2. A summary is generated.
3. The summary replaces old messages.
4. Recent messages remain unchanged.

This significantly reduces token usage while preserving important information.

---

## Compression History

Every compression operation is recorded.

Information stored includes:

- compression time
- number of folded messages
- generated summary

This can be useful for debugging and analytics.

---

## Testing

Run the tests using:

```bash
python test_conversation_manager.py
```

Example output is available in:

```
test_run_output.txt
```

---

## Use Cases

This project is useful for:

- AI Chatbots
- Voice Assistants
- Customer Support Bots
- Personal AI Assistants
- Long-running Conversations
- LLM Memory Management
- Multi-session Chat Systems

---

## Advantages

- Reduces token usage
- Preserves important information
- Improves scalability
- Works offline
- Easy to integrate
- Configurable
- Supports AI-powered summarization

---

## Future Improvements

- Semantic summarization
- Vector memory integration
- Multiple summarization strategies
- User profile extraction
- Memory ranking
- Retrieval-Augmented Generation (RAG)
- Better token estimation using tokenizer libraries

---

## License

This project is intended for educational and research purposes. Modify and extend it as needed.
