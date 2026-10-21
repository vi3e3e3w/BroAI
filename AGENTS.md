# AGENTS.md - BroAI

## Project identity
- Project name: BroAI.
- BroAI is a small character-level chatbot implemented in Python and PyTorch.
- This is not a modern large language model.
- Terminal UI banner: `Cleverbot LLM Edition`.

## Environment
- Project root: `/home/giang/Projects/ai-project`
- Virtual environment: `.venv`
- Main launcher: `ai-cli`
- Run the application with `python -m src.main` from the project root.
- Train with `python -m src.train`.
- Prepare processed learning data with `python -m src.learn`.

## Architecture
- `src/model.py`: model architecture.
- `src/dataset.py`: dataset loading and preparation utilities.
- `src/tokenizer.py`: character tokenizer and vocabulary mappings.
- `src/learn.py`: learning-data preparation.
- `src/train.py`: training loop and checkpoint creation.
- `src/generate.py`: response generation and output cleanup.
- `src/main.py`: terminal chat interface.
- `checkpoints/model.pt`: trained model checkpoint.

Inspect the current source before assuming these responsibilities or interfaces have not changed.

## Rules for code changes
1. Inspect relevant source files, tests, and `git status` before editing.
2. Preserve all existing user changes, including uncommitted and untracked files.
3. Do not run destructive Git commands, discard user changes, or delete files without explicit permission.
4. Do not commit, create branches, or push unless explicitly requested.
5. Make the smallest focused change that addresses the root cause.
6. Do not retrain the model unless explicitly requested.
7. Do not modify training data or checkpoints unless the task requires it and the user has authorized the change.
8. Do not change tokenizer vocabulary or checkpoint formats without checking compatibility with existing checkpoints.
9. Explain the expected effects and risks of changes before performing risky operations.
10. Never claim a test passed unless it was actually run and passed.

## TAG and conversation format
- The project uses `<USER>` and `<AI>` role markers in chat data.
- Keep training-data formatting and inference-time parsing consistent.
- Preserve role boundaries and prevent role markers from leaking into generated replies.
- Handle malformed or partial tags carefully without deleting ordinary text that merely resembles a tag.
- Keep output cleanup deterministic and covered by regression tests.
- Remember that the tokenizer is character-level; role markers are not automatically special tokens.

## Data and generation quality
- Investigate data boundaries, context construction, stopping conditions, and generation logic before increasing training epochs.
- Do not assume low training loss guarantees coherent responses.
- Preserve conversation boundaries where the dataset format supports them.
- Avoid unrelated changes to generation behavior while fixing a specific bug.

## Testing and validation
- Run relevant regression tests after code changes.
- Run `python -m py_compile` on modified Python files when appropriate.
- Run `git diff --check`.
- Inspect the final diff and summarize modified files.
- Report tests that could not be run and any remaining limitations honestly.

## Working style
- Explain technical decisions in clear, concise language.
- State assumptions when source code does not establish a fact.
- Do not silently expand the task scope.
- Finish with a summary of changes, tests, and outstanding issues.
