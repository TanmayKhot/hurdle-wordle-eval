# Hurdle Wordle Environment

A custom environment implementing Hurdle Wordle, a challenging variant of Wordle designed for testing LLMs (Large Language Models).

### Overview
- **Environment ID**: `hurdle-wordle`
- **Short description**: A Wordle variant that provides only counts of green/yellow letters, not their positions
- **Tags**: hurdle-wordle, wordle, word-game, reasoning, llm-testing

### Datasets
- **Primary dataset(s)**: Uses standard English 5-letter word dictionary from TextArena Wordle
- **Source links**: Integrated with TextArena framework
- **Split sizes**: Configurable (default: 2000 train, 20 eval)

### Task
- **Type**: Multi-turn interactive word guessing game
- **Parser**: XMLParser with `<think>` and `<guess>` fields
- **Rubric overview**: 
  - Exact match reward (1.0 for correct guess)
  - Partial credit based on green/yellow counts (0.2 per green, 0.1 per yellow)
  - Turn efficiency reward (1.0 / (turns + 1))
  - Format compliance reward

### Game Rules

- **Objective**: Guess a secret 5-letter word
- **Attempts**: 8 chances (compared to 6 in regular Wordle)
- **Feedback**: After each guess, you receive:
  - `greens`: Number of letters that are correct and in the correct position
  - `yellows`: Number of letters that are correct but in the wrong position
- **No Position Information**: Unlike regular Wordle, exact positions of green/yellow letters are not revealed
- **Word Validation**: All guesses must be valid 5-letter words from the game dictionary

### Key Differences from Regular Wordle

| Feature | Regular Wordle | Hurdle Wordle |
|---------|---------------|---------------|
| Feedback | Position-specific colors | Only counts |
| Attempts | 6 | 8 |
| Difficulty | Moderate | High |
| Information | Full position data | Minimal information |

### Example Gameplay

```
Secret word: PLANT (hidden)

Guess 1: [CRANE]
Feedback: greens: 2, yellows: 0

Guess 2: [AUDIO] 
Feedback: greens: 0, yellows: 1

Guess 3: [PLANT]
Feedback: greens: 5, yellows: 0
🎉 Congratulations! You guessed the word correctly!
```

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval hurdle-wordle
```

Configure model and sampling:

```bash
uv run vf-eval hurdle-wordle -m gpt-4o-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"use_think": true}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `2000` | Number of training examples |
| `num_eval_examples` | int | `20` | Number of evaluation examples |
| `use_think` | bool | `true` | Whether to use thinking step before guessing |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `check_answer_reward_func` | 1.0 if correct guess, 0.0 otherwise |
| `partial_credit_reward_func` | 0.2 × greens + 0.1 × yellows |
| `count_turns_reward_func` | Efficiency reward: 1.0 / (turns + 1) |
| `format_reward` | Compliance with XML format requirements |

### Implementation Features

- **Proper Feedback Calculation**: Handles repeated letters correctly using standard Wordle logic
- **Input Validation**: Checks word format, length, and dictionary validity  
- **Game State Management**: Tracks guess history and game progress
- **Win/Lose Conditions**: Detects wins and enforces 8-guess limit
- **LLM Integration**: Compatible with verifiers framework for testing language models

### Testing Coverage

The environment includes comprehensive unit tests covering:

- ✅ Guess with all wrong letters (greens=0, yellows=0)
- ✅ Guess with some yellows but no greens  
- ✅ Guess with repeated letters
- ✅ Winning on first attempt
- ✅ Reaching 8 attempts without success
- ✅ Input validation (format, length)
- ✅ Complex feedback scenarios

### Technical Notes

The feedback calculation follows standard Wordle logic:
1. First pass: Count exact position matches (greens) and mark positions as used
2. Second pass: Count letter matches in wrong positions (yellows) from remaining letters

This ensures proper handling of repeated letters and matches the behavior players expect from Wordle-style games.

---

## Evaluation Reports

*Evaluation reports will be automatically rendered below when available*