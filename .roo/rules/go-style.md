# Go Coding Style

## Formatting

- **gofmt** and **goimports** are mandatory — no style debates

## Design Principles

- Accept interfaces, return structs
- Keep interfaces small (1-3 methods)

## Error Handling

Always wrap errors with context:

```go
if err != nil {
    return fmt.Errorf("failed to create user: %w", err)
}
```

## Testing

- Use table-driven tests
- Use `testing` package
- Use `testify` for assertions
- Keep test functions small and focused