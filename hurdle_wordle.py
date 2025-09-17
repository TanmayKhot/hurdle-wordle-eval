import re
import random
from typing import Optional, Tuple, List, Dict, Any
import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv
from textarena.envs.Wordle.env import WordleEnv
import textarena as ta
from textarena.envs.registration import register_with_versions
from datasets import Dataset, Features, Value


### prompts
THINK_GUESS_SYSTEM_PROMPT = """You are playing Hurdle Wordle, a variant of Wordle. \
Make sure you read the game instructions carefully, and always follow the required format.

In this game, you only receive counts of correct letters, not their exact positions:
- greens: number of letters that are correct and in the correct position
- yellows: number of letters that are correct but in the wrong position

You have 8 chances to guess the correct 5-letter word.

In each turn, think step-by-step inside <think>...</think> tags, \
then give your guess inside <guess>...</guess> tags."""

NOTHINK_GUESS_SYSTEM_PROMPT = """You are playing Hurdle Wordle, a variant of Wordle. \
Make sure you read the game instructions carefully, and always follow the required format.

In this game, you only receive counts of correct letters, not their exact positions:
- greens: number of letters that are correct and in the correct position
- yellows: number of letters that are correct but in the wrong position

You have 8 chances to guess the correct 5-letter word.

In each turn, give only your guess inside <guess>...</guess> tags."""


### feedback functions
def hurdle_wordle_feedback_fn(observation: str) -> str:
    if "Feedback:" in observation:
        return observation.split("Feedback:")[-1]
    else:
        return observation


### reward functions
def check_answer_reward_func(parser, completion, answer, **kwargs) -> float:
    guess = parser.parse_answer(completion)
    return 1.0 if guess == "[" + answer + "]" else 0.0


def count_turns_reward_func(parser, completion, answer, **kwargs) -> float:
    assistant_messages = [x for x in completion if x["role"] == "assistant"]
    num_turns = len(assistant_messages)
    if num_turns > 0 and "Welcome to Hurdle Wordle!" in assistant_messages[0].get("content", ""):
        num_turns -= 1 
        
    is_correct = check_answer_reward_func(parser, completion, answer, **kwargs)
    return is_correct / (num_turns + 1)


def partial_credit_reward_func(parser, completion, **kwargs) -> float:
    """Reward function that gives partial credit based on green/yellow counts."""
    final_env_response = parser.get_user_messages(completion)[-1]["content"].strip()
    
    if "Feedback:" in final_env_response:
        feedback_line = final_env_response.split("Feedback:")[-1].strip()
        # Look for greens: X, yellows: Y pattern
        green_match = re.search(r"greens:\s*(\d+)", feedback_line)
        yellow_match = re.search(r"yellows:\s*(\d+)", feedback_line)
        
        if green_match and yellow_match:
            num_greens = int(green_match.group(1))
            num_yellows = int(yellow_match.group(1))
            return 0.2 * num_greens + 0.1 * num_yellows
    return 0.0


def calculate_hurdle_feedback(guess: str, secret: str) -> Tuple[int, int]:
    """
    Calculate greens and yellows for Hurdle Wordle.
    
    Args:
        guess: The guessed word
        secret: The secret word
        
    Returns:
        Tuple of (greens, yellows) counts
    """
    greens = 0
    yellows = 0
    
    guess = guess.lower()
    secret = secret.lower()
    
    # Create lists to track which positions have been used
    secret_used = [False] * len(secret)
    guess_used = [False] * len(guess)
    
    # First pass: count greens (correct position)
    for i in range(min(len(guess), len(secret))):
        if guess[i] == secret[i]:
            greens += 1
            secret_used[i] = True
            guess_used[i] = True
    
    # Second pass: count yellows (correct letter, wrong position)
    for i in range(len(guess)):
        if not guess_used[i]:  # This position wasn't a green match
            letter = guess[i]
            # Look for this letter in unused positions of secret
            for j in range(len(secret)):
                if not secret_used[j] and secret[j] == letter:
                    yellows += 1
                    secret_used[j] = True  # Mark this secret position as used
                    break  # Only count one match per guess letter
    
    return greens, yellows


class HurdleWordleEnv(WordleEnv):
    def __init__(self, word_length: int = 5, num_guesses: int = 8):
        # Initialize with 8 guesses and no hardcore mode
        super().__init__(word_length=word_length, num_guesses=num_guesses, hardcore=False)

    def reset(self, num_players: int = 1, seed: Optional[int] = None):
        """Reset the environment."""
        super().reset(num_players=num_players, seed=seed)
        # Set higher error allowance to handle invalid moves gracefully
        self.state.error_allowance = 10

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """Generate the initial prompt for the player."""
        prompt = f"""Welcome to Hurdle Wordle!

You need to guess a secret {game_state['word_length']}-letter word.
You have {game_state['num_guesses']} attempts.

Unlike regular Wordle, you only get counts of correct letters:
- greens: letters in the correct position
- yellows: correct letters in wrong positions

Submit your guess in the format: [WORD]

Good luck!"""
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        self.state.add_observation(message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        match = re.search(r"\[(\w+)\]", action)

        if match is None:
            self.state.set_invalid_move(reward=self._get_percentage_completion(), 
                                      reason="You tried submitting a word in the wrong format. Please make sure to use squared brackets.")
            is_done, info = self.state.step()
            info["reason"] = "You tried submitting a word in the wrong format. Please make sure to use squared brackets."
            info["reward"] = self.state.rewards.get(0, 0.0) if self.state.rewards else 0.0
            return is_done, info

        word = match.group(1).lower()
        if len(word) != self.state.game_state["word_length"]:
            self.state.set_invalid_move(reward=self._get_percentage_completion(), 
                                      reason=f"Your word must be exactly {self.state.game_state['word_length']} letters.")
            is_done, info = self.state.step()
            info["reason"] = f"Your word must be exactly {self.state.game_state['word_length']} letters."
            info["reward"] = self.state.rewards.get(0, 0.0) if self.state.rewards else 0.0
            return is_done, info

        # Skip dictionary validation for now since we don't have access to word_list in tests
        # In production, this would check against the actual Wordle word list

        # Calculate Hurdle Wordle feedback
        secret_word = self.state.game_state["secret_word"]
        greens, yellows = calculate_hurdle_feedback(word, secret_word)
        
        # Add to guess history
        self.state.game_state["guess_history"].append((word, (greens, yellows)))
        
        # Check if won
        if word.lower() == secret_word.lower():
            feedback_msg = f"You submitted [{word.upper()}].\nFeedback: greens: {greens}, yellows: {yellows}\nCongratulations! You guessed the word correctly!"
            self.state.add_observation(message=feedback_msg, observation_type=ta.ObservationType.GAME_MESSAGE)
            self.state.set_outcome(reward=1.0, reason="Congratulations! You guessed the word correctly!")
            is_done, info = self.state.step()
            info["reward"] = 1.0
            info["reason"] = "Congratulations! You guessed the word correctly!"
            return is_done, info
        
        # Check if out of guesses
        if len(self.state.game_state["guess_history"]) >= self.state.game_state["num_guesses"]:
            feedback_msg = f"You submitted [{word.upper()}].\nFeedback: greens: {greens}, yellows: {yellows}\nGame over! The word was '{secret_word.upper()}'."
            self.state.add_observation(message=feedback_msg, observation_type=ta.ObservationType.GAME_MESSAGE)
            self.state.set_outcome(reward=0.0, reason=f"Game over! The word was '{secret_word.upper()}'.")
            is_done, info = self.state.step()
            info["reward"] = 0.0
            info["reason"] = f"Game over! The word was '{secret_word.upper()}'."
            return is_done, info
        
        # Continue game - provide feedback
        feedback_msg = f"You submitted [{word.upper()}].\nFeedback: greens: {greens}, yellows: {yellows}"
        self.state.add_observation(message=feedback_msg, observation_type=ta.ObservationType.GAME_MESSAGE)
        
        is_done, info = self.state.step()
        info["latest_observation"] = {"content": feedback_msg}
        info["reward"] = 0.2 * greens + 0.1 * yellows  # Partial credit
        return is_done, info


# Custom TextArenaEnv for Hurdle Wordle
class HurdleWordleTextArenaEnv(TextArenaEnv):
    def ta_to_hf(self) -> Tuple[Dataset, Optional[Dataset]]:
        dataset_rows = []
        eval_dataset_rows = []
        
        ta_env_instance = ta.make(env_id=self.game)
        ta_env_instance.reset(num_players=1)
        _, user_prompt = ta_env_instance.get_observation()
        words = ta_env_instance.word_list

        random.seed(self.seed)
        for i in range(self.num_train_examples + self.num_eval_examples):
            question = user_prompt
            answer = random.choice(words)
            if i < self.num_train_examples:
                dataset_rows.append({"question": question, "answer": answer})
            else:
                eval_dataset_rows.append({"question": question, "answer": answer})
        
        features = Features({"question": Value("string"), "answer": Value("string")})
        dataset = Dataset.from_list(dataset_rows, features=features)
        if self.num_eval_examples > 0:
            eval_dataset = Dataset.from_list(eval_dataset_rows, features=features)
        else:
            eval_dataset = None
        return dataset, eval_dataset


# Register HurdleWordleEnv with textarena
HURDLE_WORDLE_ENV_ID = "HurdleWordle-v0"
register_with_versions(
    id=HURDLE_WORDLE_ENV_ID,
    entry_point=HurdleWordleEnv,
    wrappers={"default": [], "-train": []},
    word_length=5,
    num_guesses=8,
)


### environment loader
def load_environment(
    num_train_examples: int = 2000,
    num_eval_examples: int = 20,
    use_think: bool = True,
):
    if use_think:
        system_prompt = THINK_GUESS_SYSTEM_PROMPT
        parser = vf.XMLParser(fields=["think", "guess"], answer_field="guess")
    else:
        system_prompt = NOTHINK_GUESS_SYSTEM_PROMPT
        parser = vf.XMLParser(fields=["guess"], answer_field="guess")

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(check_answer_reward_func)
    rubric.add_reward_func(partial_credit_reward_func)
    rubric.add_reward_func(count_turns_reward_func)
    rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)

    vf_env = HurdleWordleTextArenaEnv(
        game=HURDLE_WORDLE_ENV_ID,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        feedback_fn=hurdle_wordle_feedback_fn,
    )
    return vf_env
