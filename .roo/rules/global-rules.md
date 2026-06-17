# ECC Global Rules for Roo Code

## Overview

This file merges ECC's common guidelines for use with Roo Code on OpenRouter. It provides universal principles, slash command emulation, and continuous learning hooks.

---

## Core Principles

### Immutability (CRITICAL)

ALWAYS create new objects, NEVER mutate existing ones:

```
// Pseudocode
WRONG:  modify(original, field, value) → changes original in-place
CORRECT: update(original, field, value) → returns new copy with change
```

Rationale: Immutable data prevents hidden side effects, makes debugging easier, and enables safe concurrency.

### KISS (Keep It Simple)

- Prefer the simplest solution that actually works
- Avoid premature optimization
- Optimize for clarity over cleverness

### DRY (Don't Repeat Yourself)

- Extract repeated logic into shared functions or utilities
- Avoid copy-paste implementation drift
- Introduce abstractions when repetition is real, not speculative

### YAGNI (You Aren't Gonna Need It)

- Do not build features or abstractions before they are needed
- Avoid speculative generality
- Start simple, then refactor when the pressure is real

---

## Git Workflow

### Commit Message Format

```
<type>: <description>

<optional body>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

### Pull Request Workflow

1. Analyze full commit history (not just latest commit)
2. Use `git diff [base-branch]...HEAD` to see all changes
3. Draft comprehensive PR summary
4. Include test plan with TODOs
5. Push with `-u` flag if new branch

---

## Security Guidelines

### Mandatory Security Checks

Before ANY commit:
- [ ] No hardcoded secrets (API keys, passwords, tokens)
- [ ] All user inputs validated
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (sanitized HTML)
- [ ] CSRF protection enabled
- [ ] Authentication/authorization verified
- [ ] Rate limiting on all endpoints
- [ ] Error messages don't leak sensitive data

### Secret Management

- NEVER hardcode secrets in source code
- ALWAYS use environment variables or a secret manager
- Validate that required secrets are present at startup
- Rotate any secrets that may have been exposed

---

## Testing Requirements

### Minimum Test Coverage: 80%

Test Types (ALL required):
1. **Unit Tests** - Individual functions, utilities, components
2. **Integration Tests** - API endpoints, database operations
3. **E2E Tests** - Critical user flows

### Test-Driven Development

MANDATORY workflow:
1. Write test first (RED)
2. Run test - it should FAIL
3. Write minimal implementation (GREEN)
4. Run test - it should PASS
5. Refactor (IMPROVE)
6. Verify coverage (80%+)

---

## Performance Optimization

### Model Selection Strategy

- **Haiku 4.5**: Lightweight tasks, frequent invocation
- **Sonnet 4.6**: Main development work, complex coding tasks
- **Opus 4.5**: Complex architectural decisions, deep reasoning

### Context Window Management

Avoid last 20% of context window for:
- Large-scale refactoring
- Feature implementation spanning multiple files
- Debugging complex interactions

---

## Language-Specific Rules

### TypeScript/JavaScript

- Use `interface` for object shapes that may be extended
- Use `type` for unions, intersections, tuples
- Avoid `any` in application code; use `unknown` for external input
- Use spread operator for immutable updates
- Use Zod for schema-based validation

### Python

- Use type hints for public APIs
- Follow PEP 8 style guidelines
- Use pytest for testing
- Validate all user inputs with pydantic

### Go

- Use proper error handling (no ignored errors)
- Follow standard Go formatting (gofmt)
- Use table-driven tests
- Keep functions small and focused

---

## Slash Command Emulation

Since Roo Code doesn't natively run Claude Code bash-aliases, these instructions teach how to intercept and emulate slash commands.

### /plan Command

When user types `/plan [feature description]`:

1. **Switch to ecc-planner mode** (or act as planner)
2. **Analyze requirements** and restate them clearly
3. **Identify risks** and potential blockers
4. **Create step-by-step markdown checklist** with:
   - Overview
   - Requirements
   - Architecture Changes
   - Implementation Steps (phased)
   - Testing Strategy
   - Risks & Mitigations
   - Success Criteria
5. **WAIT for user confirmation** before proceeding

### /tdd Command

When user types `/tdd [feature description]`:

1. **Switch to ecc-tdd mode** (or act as TDD specialist)
2. **Enforce Red-Green-Refactor cycle**:
   - Write test first (RED)
   - Run test - verify it FAILS
   - Write minimal implementation (GREEN)
   - Run test - verify it PASSES
   - Refactor (IMPROVE)
   - Verify 80%+ coverage
3. **Test types required**:
   - Unit tests for all public functions
   - Integration tests for API endpoints
   - E2E tests for critical user flows
4. **Edge cases to test**:
   - Null/Undefined input
   - Empty arrays/strings
   - Invalid types
   - Boundary values
   - Error paths
   - Race conditions
   - Large data
   - Special characters

### /code-review Command

When user types `/code-review [file or path]`:

1. **Switch to ecc-reviewer mode** (or act as code reviewer)
2. **Gather context** via `git diff --staged` and `git diff`
3. **Apply review checklist**:
   - **CRITICAL**: Security issues (hardcoded credentials, SQL injection, XSS, path traversal, CSRF, auth bypasses)
   - **HIGH**: Code quality (large functions, deep nesting, missing error handling, mutation patterns, console.log, missing tests)
   - **MEDIUM**: Performance (inefficient algorithms, missing caching, unoptimized images)
   - **LOW**: Best practices (TODO/FIXME without tickets, missing JSDoc, poor naming, magic numbers)
4. **Output markdown report** with:
   - Severity counts table
   - Specific findings with file paths and line numbers
   - Verdict (APPROVE/WARNING/BLOCK)

### /security-scan Command

When user types `/security-scan [path]`:

1. **Switch to ecc-security mode** (or act as security reviewer)
2. **Run security analysis**:
   - `npm audit --audit-level=high`
   - `npx eslint . --plugin security`
   - Search for hardcoded secrets
3. **OWASP Top 10 Check**:
   - Injection prevention
   - Broken Auth detection
   - Sensitive Data exposure
   - XXE prevention
   - Broken Access control
   - Security Misconfiguration
   - XSS prevention
   - Insecure Deserialization
   - Known Vulnerabilities
   - Insufficient Logging
4. **Output security report** with:
   - Security grade and score
   - Critical/high findings with exact paths
   - Remediation order

---

## Continuous Learning Observer

At the end of every successful task, you MUST write down any newly discovered project patterns, custom conventions, or architectural gotchas into `.roo/rules/learned-instincts.md`.

This file acts as a local memory bank that Roo Code will load in future sessions, enabling progressive development.

### What to Record

- Project-specific naming conventions
- Custom file organization patterns
- Unique architectural decisions
- Common error patterns and solutions
- Performance optimizations discovered
- Security considerations specific to this project
- Testing patterns that work well
- Dependency management practices

### Format

```markdown
## [Date] - [Task Description]

### Patterns Discovered
- [Pattern 1 with file reference]
- [Pattern 2 with file reference]

### Conventions
- [Convention 1]
- [Convention 2]

### Gotchas
- [Gotcha 1 and solution]
- [Gotcha 2 and solution]
```

---

## OpenRouter Token & Cache Optimization

### Prompt Management

- Keep reasoning concise. Do not regurgitate code blocks unless making changes.
- When editing files, use precise search-and-replace blocks (Roo Code diffs) rather than rewriting whole files.
- Structure all rule markdown files with clean headings and minimal verbose text.
- Static prefixes in rules enable OpenRouter's prompt caching.

### System Prompt Bloat Prevention

- Do not include large code examples in reasoning unless required
- Reference existing files instead of reproducing content
- Use bullet points and tables for scannability
- Keep instructions focused and actionable

---

## Agent Orchestration

### Available ECC Modes

| Mode | Purpose | When to Use |
|------|---------|-------------|
| ecc-planner | Implementation planning | Complex features, refactoring |
| ecc-architect | System design | Architectural decisions |
| ecc-tdd | Test-driven development | New features, bug fixes |
| ecc-reviewer | Code review | After writing code |
| ecc-security | Security analysis | Before commits, auth changes |

### Mode Switching

Use the mode selector in VS Code UI or request mode switch for specialized tasks. Each mode has restricted tool permissions to save tokens and prevent unnecessary file writes.