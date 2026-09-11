"""Progress bar management with granular control."""

from enum import IntEnum
from typing import Any, Iterable, Optional

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ProgressLevel(IntEnum):
    """
    Progress verbosity levels.

    QUIET: No progress bars (silent mode)
    NORMAL: Show high-level progress only (benchmarks, runs, checks)
    VERBOSE: Show all progress including per-question iterations
    """

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2


class ProgressManager:
    """
    Centralized progress bar management.

    Allows granular control over which progress bars are shown based
    on the configured verbosity level.
    """

    def __init__(self, level: ProgressLevel = ProgressLevel.NORMAL):
        """
        Initialize progress manager.

        Args:
            level: Progress verbosity level
        """
        self.level = level

    def progress_bar(
        self,
        iterable: Iterable,
        desc: str,
        required_level: ProgressLevel = ProgressLevel.NORMAL,
        **kwargs,
    ) -> Iterable:
        """
        Create progress bar only if configured level permits.

        Args:
            iterable: Iterable to wrap with progress bar
            desc: Description for the progress bar
            required_level: Minimum level required to show this progress bar
            **kwargs: Additional arguments passed to tqdm

        Returns:
            tqdm-wrapped iterable if level permits, otherwise plain iterable

        Examples:
            >>> # Show only at VERBOSE level (hidden by default)
            >>> for item in progress_bar(items, "Processing", ProgressLevel.VERBOSE):
            ...     process(item)
            >>>
            >>> # Show at NORMAL and VERBOSE levels (visible by default)
            >>> for batch in progress_bar(batches, "Batches", ProgressLevel.NORMAL):
            ...     process_batch(batch)
        """
        if not HAS_TQDM:
            # tqdm not available, return plain iterable
            return iterable

        if self.level >= required_level:
            return tqdm(iterable, desc=desc, **kwargs)
        else:
            return iterable


# Global progress manager instance
_manager = ProgressManager()


def set_progress_level(level: ProgressLevel) -> None:
    """
    Set global progress verbosity level.

    Args:
        level: Progress level to set

    Example:
        >>> from bencheck.core.progress import set_progress_level, ProgressLevel
        >>> set_progress_level(ProgressLevel.QUIET)  # Disable all progress bars
    """
    global _manager
    _manager = ProgressManager(level)


def get_progress_level() -> ProgressLevel:
    """
    Get current progress verbosity level.

    Returns:
        Current progress level
    """
    return _manager.level


def progress_bar(
    iterable: Iterable, desc: str, required_level: ProgressLevel = ProgressLevel.NORMAL, **kwargs
) -> Iterable:
    """
    Create progress bar using global manager.

    This is the main function to use for creating progress bars throughout
    the codebase.

    Args:
        iterable: Iterable to wrap
        desc: Description for the progress bar
        required_level: Minimum level required to show this bar
        **kwargs: Additional tqdm arguments (unit, total, etc.)

    Returns:
        Progress-bar-wrapped iterable or plain iterable

    Example:
        >>> from bencheck.core.progress import progress_bar, ProgressLevel
        >>>
        >>> # High-level progress (shown at NORMAL and VERBOSE)
        >>> for run in progress_bar(range(5), "Runs", ProgressLevel.NORMAL):
        ...     process_run(run)
        >>>
        >>> # Detailed progress (shown only at VERBOSE)
        >>> for q in progress_bar(questions, "Questions", ProgressLevel.VERBOSE):
        ...     process_question(q)
    """
    return _manager.progress_bar(iterable, desc, required_level, **kwargs)
