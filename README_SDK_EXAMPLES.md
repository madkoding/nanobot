# Nanobot SDK Usage Examples

This document demonstrates how to use the Nanobot Python SDK programmatically.

## Installation

```bash
pip install nanobot-ai
```

## Quick Start

### Basic Usage

```python
import asyncio
from nanobot import Nanobot, ValidationError

async def main():
    # Create instance from config
    bot = Nanobot.from_config()
    
    # Run a simple query
    result = await bot.run("Summarize this repository")
    print(result.content)
    
    # Don't forget to close resources
    await bot.aclose()

asyncio.run(main())
```

### Using Context Manager

```python
import asyncio
from nanobot import Nanobot

async def main():
    async with Nanobot.from_config() as bot:
        result = await bot.run("What files are in this project?")
        print(result.content)
        # Resources automatically cleaned up

asyncio.run(main())
```

## Advanced Features

### Session Management

Different session keys maintain independent conversation histories:

```python
async with Nanobot.from_config() as bot:
    # Session 1: Code review
    result1 = await bot.run(
        "Review the main.py file",
        session_key="code-review-session"
    )
    
    # Session 2: Documentation
    result2 = await bot.run(
        "Write documentation for the API",
        session_key="docs-session"
    )
```

### Streaming Responses

Get real-time streaming events:

```python
async with Nanobot.from_config() as bot:
    async for event in bot.stream("Explain quantum computing"):
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "run_completed":
            print(f"\n\nTotal tokens: {event.usage['total_tokens']}")
```

### Handling Validation Errors

The SDK validates all inputs to prevent security issues:

```python
from nanobot import Nanobot, ValidationError

async with Nanobot.from_config() as bot:
    try:
        # This will raise ValidationError - invalid characters
        await bot.run(
            "Test message",
            session_key="invalid@key!"  # Only alphanumeric, _, :, - allowed
        )
    except ValidationError as e:
        print(f"Validation failed: {e}")
    
    try:
        # This will raise ValidationError - message too long (>50k chars)
        await bot.run("a" * 60000)
    except ValidationError as e:
        print(f"Message too long: {e}")
    
    try:
        # This will raise ValidationError - path traversal attempt
        await bot.run(
            "Analyze this file",
            media=["../../../etc/passwd"]
        )
    except ValidationError as e:
        print(f"Security violation: {e}")
```

### Model Selection Per Request

Override the default model for specific requests:

```python
async with Nanobot.from_config() as bot:
    # Use default model
    result1 = await bot.run("Simple question")
    
    # Override with specific model
    result2 = await bot.run(
        "Complex reasoning task",
        model="claude-sonnet-4-20250514"
    )
    
    # Or use model preset
    result3 = await bot.run(
        "Fast response needed",
        model_preset="fast"
    )
```

### Attaching Media Files

```python
async with Nanobot.from_config() as bot:
    result = await bot.run(
        "Analyze this image",
        media=["/path/to/image.png", "/path/to/document.pdf"]
    )
```

### Ephemeral Sessions

Don't persist conversation history:

```python
async with Nanobot.from_config() as bot:
    result = await bot.run(
        "Temporary query",
        ephemeral=True  # Won't be saved to session history
    )
```

### Custom Hooks

Add custom lifecycle hooks:

```python
from nanobot.agent.hook import AgentHook

class MyCustomHook(AgentHook):
    async def on_turn_start(self, turn):
        print(f"Starting turn: {turn.id}")
    
    async def on_turn_end(self, turn, result):
        print(f"Completed turn: {turn.id}")

async with Nanobot.from_config() as bot:
    result = await bot.run(
        "Process with hooks",
        hooks=[MyCustomHook()]
    )
```

## Input Validation Reference

### Session Key
- **Max length**: 64 characters
- **Allowed characters**: Alphanumeric (a-z, A-Z, 0-9), underscore (_), colon (:), hyphen (-)
- **Examples**: `"default"`, `"user-123"`, `"session:abc_123"`

### Chat ID / Sender ID / Channel
- **Max length**: 128 characters (64 for channel)
- **Allowed characters**: Alphanumeric, underscore (_), hyphen (-)
- **Examples**: `"direct"`, `"channel-general"`, `"user_admin"`

### Message
- **Max length**: 50,000 characters
- **Sanitization**: Leading/trailing whitespace stripped
- **Empty messages**: Allowed (will be passed through)

### Media Paths
- **Max length**: 512 characters per path
- **Security**: Path traversal (`..`) is blocked
- **Resolution**: Paths are resolved to absolute paths

## Error Handling

```python
from nanobot import Nanobot, ValidationError

async def safe_run(bot: Nanobot, message: str):
    try:
        result = await bot.run(message)
        return result.content
    except ValidationError as e:
        # Input validation failed
        print(f"Invalid input: {e}")
        return None
    except Exception as e:
        # Other runtime errors (API failures, etc.)
        print(f"Runtime error: {e}")
        return None
```

## Best Practices

1. **Always use context managers** or call `aclose()` to release resources
2. **Use meaningful session keys** to organize conversations
3. **Validate user inputs** before passing to SDK (though SDK does validate)
4. **Handle ValidationError** separately from other exceptions
5. **Use ephemeral sessions** for temporary queries
6. **Stream large responses** instead of waiting for completion

## Migration Guide

### From Previous Versions

If you're upgrading from a version before these changes:

**Breaking Changes:**
- Session keys with special characters now raise `ValidationError`
- Messages over 50k characters are rejected
- Media paths with `..` components are blocked

**Migration Steps:**
1. Update session keys to use only allowed characters
2. Truncate or split long messages
3. Sanitize file paths before passing to SDK

```python
# Old code (might break)
await bot.run("msg", session_key="user@email")

# New code (safe)
from nanobot.utils.validation import validate_session_key
safe_key = validate_session_key("user_email")  # Convert @ to _
await bot.run("msg", session_key=safe_key)
```
