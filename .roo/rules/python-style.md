# Python Coding Style

## Standards

- Follow **PEP 8** conventions
- Use **type annotations** on all function signatures

## Immutability

Prefer immutable data structures:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class User:
    name: str
    email: str

from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
```

## Formatting

- **black** for code formatting
- **isort** for import sorting
- **ruff** for linting

## Error Handling

Always handle errors explicitly:

```python
def get_error_message(error: Exception) -> str:
    return str(error)

def load_user(user_id: str) -> User:
    try:
        result = risky_operation(user_id)
        return result
    except Exception as error:
        logger.error("Operation failed", error)
        raise
```

## Input Validation

Use pydantic for schema-based validation:

```python
from pydantic import BaseModel, EmailStr

class UserInput(BaseModel):
    email: EmailStr
    age: int

validated = UserInput.model_validate(input)
```

## Testing

- Use pytest for testing
- Use type hints for public APIs
- Follow PEP 8 style guidelines