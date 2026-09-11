"""Configuration utilities for BenCheck."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class Config:
    """Configuration manager for BenCheck runs."""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize from config dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict

        # NEW: Experiment matrix support
        self.experiment = config_dict.get("experiment")
        if self.experiment:
            # Experiment matrix mode
            self.n_runs = self.experiment.get("n_runs", 1)
            self.scoring_modes = self.experiment.get("scoring_modes", ["log_likelihood"])
            self.seed = self.experiment.get("seed")
            # In experiment mode, benchmarks/models are under "experiment"
            self.benchmarks = self.experiment.get("benchmarks")
            self.models_list = self.experiment.get("models", [])
            self.model = None  # No single model in experiment mode
            self.checks = self.experiment.get("checks", [])
            self.benchmark = None
        else:
            # Regular mode
            self.n_runs = config_dict.get("n_runs", 1)
            self.scoring_modes = None
            self.seed = config_dict.get("seed")
            self.models_list = None

            # Support both "benchmark" (single) and "benchmarks" (multiple)
            if "benchmarks" in config_dict:
                self.benchmarks = config_dict["benchmarks"]
                self.benchmark = None  # Multi-benchmark mode
            else:
                self.benchmark = config_dict.get("benchmark", {})
                self.benchmarks = None  # Single-benchmark mode

            self.model = config_dict.get("model")
            self.checks = config_dict.get("checks", [])

        self.output = config_dict.get("output", {})

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from file.

        Supports JSON and YAML formats.

        Args:
            path: Path to configuration file

        Returns:
            Config instance

        Raises:
            ValueError: If file format is unsupported
            ImportError: If YAML file but pyyaml not installed
        """
        path = Path(path)

        if path.suffix == ".json":
            with open(path) as f:
                config_dict = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise ImportError(
                    "pyyaml required for YAML configs. Install with: pip install pyyaml"
                )
            with open(path) as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        return cls(config_dict)

    def is_multi_benchmark(self) -> bool:
        """Check if config uses multiple benchmarks."""
        return self.benchmarks is not None

    def get_benchmarks_list(self) -> List[Dict[str, Any]]:
        """
        Get list of benchmark configs.

        Returns:
            List of benchmark dicts (always a list, even for single benchmark)
        """
        if self.is_multi_benchmark():
            return self.benchmarks
        else:
            return [self.benchmark] if self.benchmark else []

    def get_adapter(self, benchmark_config: Optional[Dict[str, Any]] = None):
        """
        Create adapter from config.

        Args:
            benchmark_config: Specific benchmark config (for multi-benchmark mode).
                            If None, uses self.benchmark (single-benchmark mode).

        Returns:
            Adapter instance
        """
        from bencheck import adapters

        # Use provided benchmark_config or fallback to self.benchmark
        bench_cfg = benchmark_config if benchmark_config is not None else self.benchmark

        if not bench_cfg:
            raise ValueError("No benchmark configuration provided")

        adapter_name = bench_cfg.get("adapter")
        if not adapter_name:
            raise ValueError("benchmark.adapter required in config")

        adapter_class = getattr(adapters, adapter_name)
        params = bench_cfg.get("params", {})

        return adapter_class(**params)

    def _create_model_from_config(self, model_config: Optional[Dict[str, Any]]):
        """
        Create model from model config dict.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Model instance or None
        """
        if not model_config:
            return None

        from bencheck import models

        model_type = model_config.get("type")
        if not model_type:
            raise ValueError("model.type required when model is specified")

        model_name = model_config.get("name")
        params = dict(model_config.get("params", {}))

        if model_type == "vllm":
            return models.VLLMModel(model_name=model_name, **params)
        elif model_type == "gemini":
            return models.GeminiModel(model_name=model_name, **params)
        elif model_type == "openai":
            return models.OpenAIModel(model_name=model_name, **params)
        elif model_type == "dummy":
            return models.DummyModel()
        elif model_type in {"gector", "gector_grammar"}:
            model_paths_param = params.pop("model_paths", None)

            combined_model_paths = []
            if model_name:
                combined_model_paths.append(model_name)
            if model_paths_param:
                if isinstance(model_paths_param, str):
                    combined_model_paths.append(model_paths_param)
                else:
                    combined_model_paths.extend(model_paths_param)

            if not combined_model_paths:
                raise ValueError("model.name or params.model_paths required for GECToRGrammarModel")

            vocab_path = params.pop("vocab_path", None)
            if vocab_path is None:
                raise ValueError("params.vocab_path is required for GECToRGrammarModel")

            return models.GECToRGrammarModel(
                model_paths=combined_model_paths,
                vocab_path=vocab_path,
                **params,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_model(self):
        """
        Create global model from config.

        Returns:
            Model instance or None for model-free checks
        """
        return self._create_model_from_config(self.model)

    def get_checks(self, default_model=None):
        """
        Create checks from config.

        Each check can have its own model config, or use the default_model.
        Format:
            checks:
              - name: "CheckName"
                model: null  # model-free
                params: {}
              - name: "OtherCheck"
                model:  # per-check model (overrides default)
                  type: "vllm"
                  name: "model-name"
                params: {}
              - name: "ThirdCheck"
                # No model key - uses default_model
                params: {}

        Args:
            default_model: Default model instance for checks without specific model

        Returns:
            List of check instances
        """
        import inspect

        from bencheck import checks

        check_instances = []

        for check_config in self.checks:
            check_name = check_config.get("name")
            if not check_name:
                raise ValueError("check.name required")

            check_class = getattr(checks, check_name)
            params = check_config.get("params", {})

            # Determine which model to use for this check
            model_to_use = default_model  # Default: use global model

            if "model" in check_config:
                # Check has its own model config
                check_model_config = check_config["model"]
                if check_model_config is None:
                    # Explicitly null - model-free check
                    model_to_use = None
                else:
                    # Create per-check model
                    model_to_use = self._create_model_from_config(check_model_config)

            # Add model if check requires it
            if hasattr(check_class, "__init__"):
                sig = inspect.signature(check_class.__init__)
                if "model" in sig.parameters:
                    params["model"] = model_to_use

            check_instances.append(check_class(**params))

        return check_instances

    def get_output_dir(self) -> Path:
        """Get output directory path."""
        output_dir = self.output.get("dir", "results")
        return Path(output_dir)

    def get_output_format(self) -> str:
        """Get output format (json or csv)."""
        return self.output.get("format", "json")

    def should_save_questions(self) -> bool:
        """Check if individual questions should be saved."""
        return self.output.get("save_questions", True)

    def is_experiment_matrix(self) -> bool:
        """Check if config is experiment matrix type."""
        return self.experiment is not None and "benchmarks" in self.experiment

    def generate_experiment_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all experiment combinations from matrix config.

        For experiment matrix configs, generates all combinations of:
        - benchmarks × models × scoring_modes × runs

        Returns:
            List of dicts, each containing:
                - benchmark_name, benchmark_config
                - model_name, model_config
                - scoring_mode (if applicable)
                - run_index
                - seed

        Example:
            >>> config = Config.from_file("experiments/full_suite.yaml")
            >>> combos = config.generate_experiment_combinations()
            >>> len(combos)  # 2 benchmarks × 2 models × 2 modes × 5 runs
            40
        """
        if not self.is_experiment_matrix():
            return []

        from .scoring_mode import get_supported_modes

        combinations = []
        benchmarks = self.experiment.get("benchmarks", [])
        models = self.models_list or []
        n_runs = self.n_runs
        scoring_modes = self.scoring_modes or []

        for bench_cfg in benchmarks:
            for model_cfg in models:
                # Create model instance to detect supported modes
                try:
                    model_instance = self._create_model_from_config(model_cfg)
                    supported_modes = get_supported_modes(model_instance)
                except Exception:
                    # If model creation fails, assume it supports what was requested
                    supported_modes = scoring_modes

                # Filter modes to only those supported by this model
                applicable_modes = [m for m in scoring_modes if m in supported_modes]

                # If no modes specified or no modes supported, run without mode
                if not applicable_modes:
                    applicable_modes = [None]

                for mode in applicable_modes:
                    for run_idx in range(n_runs):
                        combinations.append(
                            {
                                "benchmark_name": bench_cfg.get("name", "benchmark"),
                                "benchmark_config": bench_cfg,
                                "model_name": model_cfg.get("name", model_cfg.get("type", "model")),
                                "model_config": model_cfg,
                                "scoring_mode": mode,
                                "run_index": run_idx,
                                "seed": (self.seed + run_idx) if self.seed is not None else None,
                            }
                        )

        return combinations
