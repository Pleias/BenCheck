"""Specialized MCQ adapters for popular benchmarks.

Each adapter inherits from BaseMCQAdapter and defines dataset-specific configuration.
"""

from .base_mcq import BaseMCQAdapter


class MMLUAdapter(BaseMCQAdapter):
    """
    Adapter for MMLU (Massive Multitask Language Understanding) benchmark.

    Dataset: cais/mmlu
    Format:
        - question: The question text
        - choices: List of 4 answer options [A, B, C, D]
        - answer: Correct answer index (0-3) or letter (A-D)

    Example:
        >>> adapter = MMLUAdapter(split="test")
        >>> questions = adapter.load()
    """

    dataset_name = "cais/mmlu"
    option_key = "choices"
    label_key = "answer"
    context_key = "question"

    def __init__(self, split: str = "test", config_name: str = "all"):
        """
        Initialize MMLU adapter.

        Args:
            split: Dataset split (default: "test")
            config_name: MMLU subject or subset (default: "all")
        """
        super().__init__(split=split)
        self.config_name = config_name


class TruthfulQAAdapter(BaseMCQAdapter):
    """
    Adapter for TruthfulQA benchmark (multiple-choice format).

    Dataset: truthful_qa
    Config: multiple_choice
    Format:
        - question: The question text
        - mc1_targets: Dict with "choices" (list of options) and "labels" (list of 0/1)
        - mc2_targets: Similar format for mc2

    Example:
        >>> adapter = TruthfulQAAdapter(mc_type="mc1")
        >>> questions = adapter.load()
    """

    dataset_name = "truthful_qa"
    config_name = "multiple_choice"
    context_key = "question"

    def __init__(self, split: str = "validation", mc_type: str = "mc1"):
        """
        Initialize TruthfulQA adapter.

        Args:
            split: Dataset split (default: "validation")
            mc_type: Multiple choice type - "mc1" or "mc2" (default: "mc1")
        """
        super().__init__(split=split)
        self.mc_type = mc_type
        # Set option and label keys based on mc_type
        self.option_key = f"{mc_type}_targets"
        self.label_key = f"{mc_type}_targets"


class BigBenchAdapter(BaseMCQAdapter):
    """
    Adapter for BigBench tasks.

    Dataset: google/bigbench
    Config: Task name (e.g., "causal_judgment", "formal_fallacies")
    Format:
        - inputs: Question text (may include inline "choice:" strings)
        - multiple_choice_targets: List of choice options
        - multiple_choice_scores: Binary array (1=correct, 0=incorrect)
        - targets: List of correct answer texts (for reference)

    Supports both single-choice (one 1 in scores) and MCQ (multiple 1s).

    Example:
        >>> adapter = BigBenchAdapter(task_name="causal_judgment")
        >>> questions = adapter.load()
    """

    dataset_name = "google/bigbench"
    context_key = "inputs"
    option_key = "multiple_choice_targets"
    label_key = "multiple_choice_scores"

    def __init__(self, task_name: str, split: str = "validation"):
        """
        Initialize BigBench adapter.

        Args:
            task_name: BigBench task name (e.g., "causal_judgment")
            split: Dataset split (default: "validation")
        """
        super().__init__(split=split, config_name=task_name)
        self.task_name = task_name

    def _convert_record(self, record, idx):
        """
        Custom conversion for BigBench format.

        BigBench uses binary scores array (multiple_choice_scores) where:
        - 1 indicates correct answer
        - 0 indicates incorrect answer

        Need to override to properly detect single-choice vs MCQ based on
        the count of 1s in the scores array, not the array length.
        """
        from ..utils import mcq_helpers

        # Validate required fields
        if "multiple_choice_targets" not in record:
            return None
        if "multiple_choice_scores" not in record:
            return None

        options = record["multiple_choice_targets"]
        scores = record["multiple_choice_scores"]

        # Validate data quality
        if not options or not scores:
            return None

        if len(options) != len(scores):
            # Mismatched lengths - skip
            return None

        # Count correct answers (number of 1s in scores)
        num_correct = sum(1 for s in scores if int(s) == 1)

        if num_correct == 0:
            # No correct answer - skip this record
            return None

        # Temporarily modify record to use dict format like TruthfulQA
        # This allows BaseMCQAdapter to correctly detect MCQ vs single-choice
        modified_record = dict(record)
        modified_record[self.label_key] = {
            "labels": scores  # Binary array format
        }

        # Call parent conversion
        return super()._convert_record(modified_record, idx)


class ARC_Adapter(BaseMCQAdapter):
    """
    Adapter for AI2 Reasoning Challenge (ARC) benchmark.

    Dataset: ai2_arc
    Config: ARC-Easy or ARC-Challenge
    Format:
        - question: The question text
        - choices: Dict with "text" (list of options) and "label" (list of letters)
        - answerKey: Correct answer letter (A-E)

    Example:
        >>> adapter = ARC_Adapter(challenge_type="ARC-Challenge")
        >>> questions = adapter.load()
    """

    dataset_name = "ai2_arc"
    context_key = "question"
    label_key = "answerKey"

    def __init__(self, challenge_type: str = "ARC-Challenge", split: str = "validation"):
        """
        Initialize ARC adapter.

        Args:
            challenge_type: "ARC-Easy" or "ARC-Challenge" (default: "ARC-Challenge")
            split: Dataset split (default: "validation")
        """
        super().__init__(split=split, config_name=challenge_type)

    def _convert_record(self, record, idx):
        """Custom conversion for ARC format (choices dict with text/label)."""
        # Extract choices text from nested dict
        if "choices" in record and isinstance(record["choices"], dict):
            if "text" in record["choices"]:
                # Temporarily inject flattened choices for helper to process
                modified_record = dict(record)
                modified_record["_choices_text"] = record["choices"]["text"]
                # Use _choices_text as option_key for this conversion
                original_option_key = self.option_key
                self.option_key = "_choices_text"
                result = super()._convert_record(modified_record, idx)
                self.option_key = original_option_key
                return result

        # Fallback to standard conversion
        return super()._convert_record(record, idx)


class WinograndeAdapter(BaseMCQAdapter):
    """
    Adapter for Winogrande benchmark.

    Dataset: winogrande
    Config: winogrande_xl (or other sizes)
    Format:
        - sentence: The sentence with underscore placeholder
        - option1, option2: The two options
        - answer: "1" or "2" (1-based, will be converted to 0-based)

    Example:
        >>> adapter = WinograndeAdapter()
        >>> questions = adapter.load()
    """

    dataset_name = "winogrande"
    config_name = "winogrande_xl"
    context_key = "sentence"
    label_key = "answer"

    def _convert_record(self, record, idx):
        """Custom conversion for Winogrande format (option1, option2)."""
        # Build choices list from option1 and option2
        if "option1" in record and "option2" in record:
            modified_record = dict(record)
            modified_record["_choices"] = [record["option1"], record["option2"]]

            # Convert 1-based answer ("1", "2") to 0-based (0, 1)
            if "answer" in record:
                answer_str = str(record["answer"])
                if answer_str in ["1", "2"]:
                    modified_record["answer"] = int(answer_str) - 1  # Convert to 0-based

            # Use _choices as option_key
            original_option_key = self.option_key
            self.option_key = "_choices"
            result = super()._convert_record(modified_record, idx)
            self.option_key = original_option_key
            return result

        # Fallback to standard conversion
        return super()._convert_record(record, idx)


class CommonsenseQAAdapter(BaseMCQAdapter):
    """
    Adapter for CommonsenseQA benchmark.

    Dataset: commonsense_qa
    Format:
        - question: The question text
        - choices: Dict with "text" (list of options) and "label" (list of letters)
        - answerKey: Correct answer letter (A-E)

    Example:
        >>> adapter = CommonsenseQAAdapter()
        >>> questions = adapter.load()
    """

    dataset_name = "commonsense_qa"
    context_key = "question"
    label_key = "answerKey"

    def _convert_record(self, record, idx):
        """Custom conversion for CommonsenseQA format (similar to ARC)."""
        # Extract choices text from nested dict
        if "choices" in record and isinstance(record["choices"], dict):
            if "text" in record["choices"]:
                modified_record = dict(record)
                modified_record["_choices_text"] = record["choices"]["text"]
                original_option_key = self.option_key
                self.option_key = "_choices_text"
                result = super()._convert_record(modified_record, idx)
                self.option_key = original_option_key
                return result

        return super()._convert_record(record, idx)


class PIQAAdapter(BaseMCQAdapter):
    """
    Adapter for PIQA (Physical Interaction QA) benchmark.

    Dataset: regisss/piqa (Parquet format, no script)
    Format:
        - goal: The question/goal
        - sol1: First solution
        - sol2: Second solution
        - label: Correct answer (0 or 1, already 0-based)

    Example:
        >>> adapter = PIQAAdapter()
        >>> questions = adapter.load()
    """

    dataset_name = "regisss/piqa"
    context_key = "goal"
    label_key = "label"

    def _convert_record(self, record, idx):
        """Custom conversion for PIQA format (goal + sol1/sol2)."""
        # Build choices list from sol1 and sol2
        if "sol1" in record and "sol2" in record:
            modified_record = dict(record)
            modified_record["_choices"] = [record["sol1"], record["sol2"]]

            # Label is already 0-based (0 or 1)
            # No conversion needed

            # Use _choices as option_key
            original_option_key = self.option_key
            self.option_key = "_choices"
            result = super()._convert_record(modified_record, idx)
            self.option_key = original_option_key
            return result

        # Fallback to standard conversion
        return super()._convert_record(record, idx)


class GlobalPIQAAdapter(BaseMCQAdapter):
    """
    Adapter for Global PIQA (multilingual PIQA) benchmark.

    Dataset: mrlbenchmarks/global-piqa-nonparallel
    Default subset: eng_latn (English Latin script)
    Format: Same as PIQA (goal, sol1, sol2, label)

    This is a multilingual version of PIQA with questions translated
    to multiple languages. By default, only English (eng_latn) is loaded.

    Example:
        >>> adapter = GlobalPIQAAdapter()  # loads eng_latn
        >>> questions = adapter.load()
        >>>
        >>> # Load different language subset
        >>> adapter = GlobalPIQAAdapter(subset="fra_latn")
        >>> questions = adapter.load()
    """

    dataset_name = "mrlbenchmarks/global-piqa-nonparallel"
    context_key = "prompt"  # Global PIQA uses "prompt" not "goal"
    label_key = "label"

    def __init__(self, subset: str = "eng_latn", split: str = "test"):  # Global PIQA only has "test" split
        """
        Initialize Global PIQA adapter.

        Args:
            subset: Language subset (default: "eng_latn" for English)
            split: Dataset split (default: "test" - only split available)
        """
        super().__init__(split=split, config_name=subset)
        self.subset = subset

    def _convert_record(self, record, idx):
        """Custom conversion for Global PIQA format (prompt + solution0/solution1)."""
        # Global PIQA uses different field names than regular PIQA
        required_fields = ["prompt", "solution0", "solution1", "label"]
        for field in required_fields:
            if field not in record:
                # Missing field - skip this record
                return None

        # Validate fields are not empty strings
        goal = record.get("prompt", "")  # "prompt" instead of "goal"
        sol1 = record.get("solution0", "")  # "solution0" instead of "sol1"
        sol2 = record.get("solution1", "")  # "solution1" instead of "sol2"

        if not goal or not isinstance(goal, str) or not goal.strip():
            # Empty or invalid goal - skip this record
            return None

        if not sol1 or not isinstance(sol1, str) or not sol1.strip():
            # Empty or invalid sol1 - skip this record
            return None

        if not sol2 or not isinstance(sol2, str) or not sol2.strip():
            # Empty or invalid sol2 - skip this record
            return None

        # Build choices list from sol1 and sol2
        modified_record = dict(record)
        modified_record["_choices"] = [sol1, sol2]

        # Label is already 0-based (0 or 1)
        # No conversion needed

        # Use _choices as option_key
        original_option_key = self.option_key
        self.option_key = "_choices"
        result = super()._convert_record(modified_record, idx)
        self.option_key = original_option_key
        return result
