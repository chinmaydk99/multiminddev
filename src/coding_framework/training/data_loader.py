"""
Training data loader for coding problems.

Handles loading, validation, and preprocessing of training data
for the VERL reinforcement learning pipeline.
"""

import json
import random
from pathlib import Path
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field, validator


class TrainingProblem(BaseModel):
    """Data model for a training problem."""

    id: str = Field(..., description="Unique problem identifier")
    problem: str = Field(..., description="Problem description")
    difficulty: str = Field(default="medium", description="Problem difficulty level")
    test_cases: list[dict[str, Any]] = Field(..., description="Test cases for validation")
    solution: Optional[str] = Field(None, description="Reference solution")
    tags: list[str] = Field(default_factory=list, description="Problem tags/categories")
    language: str = Field(default="python", description="Programming language")

    @validator("difficulty")
    def validate_difficulty(cls, v):
        if v not in ["easy", "medium", "hard"]:
            raise ValueError("Difficulty must be easy, medium, or hard")
        return v

    @validator("test_cases")
    def validate_test_cases(cls, v):
        if not v:
            raise ValueError("At least one test case is required")

        for i, test_case in enumerate(v):
            if "input" not in test_case or "expected_output" not in test_case:
                raise ValueError(f"Test case {i} must have 'input' and 'expected_output' fields")
        return v


class TrainingDataLoader:
    """
    Training data loader for coding problems.

    Handles loading JSON training data, validation, preprocessing,
    and train/validation splitting for reinforcement learning.
    """

    def __init__(self, data_path: str, validation_split: float = 0.2):
        """
        Initialize the training data loader.

        Args:
            data_path: Path to training data (file or directory)
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
        """
        self.data_path = Path(data_path)
        self.validation_split = max(0.0, min(1.0, validation_split))
        self.logger = structlog.get_logger(component="data_loader")

        # Cached data
        self._problems: Optional[list[TrainingProblem]] = None
        self._train_data: Optional[list[dict[str, Any]]] = None
        self._validation_data: Optional[list[dict[str, Any]]] = None

        self.logger.info(
            "Training data loader initialized",
            data_path=str(self.data_path),
            validation_split=self.validation_split,
        )

    async def load_problems(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Load and split training problems into train/validation sets.

        Returns:
            Tuple of (training_data, validation_data)
        """
        if self._train_data is not None and self._validation_data is not None:
            return self._train_data, self._validation_data

        try:
            # Load all problems
            problems = await self._load_all_problems()

            # Validate problems
            validated_problems = await self._validate_problems(problems)

            # Split into train/validation
            train_problems, val_problems = self._split_data(validated_problems)

            # Convert to dict format for training
            self._train_data = [self._problem_to_dict(p) for p in train_problems]
            self._validation_data = [self._problem_to_dict(p) for p in val_problems]

            self.logger.info(
                "Training data loaded successfully",
                total_problems=len(validated_problems),
                training_problems=len(self._train_data),
                validation_problems=len(self._validation_data),
            )

            return self._train_data, self._validation_data

        except Exception as e:
            self.logger.error(f"Failed to load training data: {e}")
            raise

    async def _load_all_problems(self) -> list[dict[str, Any]]:
        """
        Load all problems from data path.

        Returns:
            List of raw problem dictionaries
        """
        problems = []

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        if self.data_path.is_file():
            # Single file
            problems.extend(await self._load_from_file(self.data_path))

        elif self.data_path.is_dir():
            # Directory - load all JSON files
            json_files = list(self.data_path.glob("*.json"))
            if not json_files:
                raise ValueError(f"No JSON files found in directory: {self.data_path}")

            for json_file in json_files:
                try:
                    file_problems = await self._load_from_file(json_file)
                    problems.extend(file_problems)
                    self.logger.debug(f"Loaded {len(file_problems)} problems from {json_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {json_file}: {e}")
                    continue

        else:
            raise ValueError(f"Data path must be a file or directory: {self.data_path}")

        if not problems:
            raise ValueError("No problems loaded from data path")

        return problems

    async def _load_from_file(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Load problems from a single JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of problem dictionaries
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if "problems" in data:
                    return data["problems"]
                else:
                    return [data]  # Single problem
            else:
                raise ValueError(f"Invalid JSON structure in {file_path}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading {file_path}: {e}")

    async def _validate_problems(self, problems: list[dict[str, Any]]) -> list[TrainingProblem]:
        """
        Validate and convert raw problems to TrainingProblem objects.

        Args:
            problems: List of raw problem dictionaries

        Returns:
            List of validated TrainingProblem objects
        """
        validated_problems = []

        for i, problem_data in enumerate(problems):
            try:
                # Add default ID if missing
                if "id" not in problem_data:
                    problem_data["id"] = f"problem_{i:04d}"

                # Validate and create TrainingProblem
                problem = TrainingProblem(**problem_data)
                validated_problems.append(problem)

            except Exception as e:
                self.logger.warning(
                    f"Invalid problem at index {i}: {e}",
                    problem_data=problem_data,
                )
                continue

        if not validated_problems:
            raise ValueError("No valid problems found after validation")

        self.logger.info(
            f"Validated {len(validated_problems)} problems "
            f"({len(problems) - len(validated_problems)} invalid)"
        )

        return validated_problems

    def _split_data(
        self,
        problems: list[TrainingProblem],
    ) -> tuple[list[TrainingProblem], list[TrainingProblem]]:
        """
        Split problems into training and validation sets.

        Args:
            problems: List of validated problems

        Returns:
            Tuple of (training_problems, validation_problems)
        """
        if self.validation_split == 0.0:
            return problems, []

        # Shuffle problems for random split
        shuffled_problems = problems.copy()
        random.shuffle(shuffled_problems)

        # Calculate split point
        total_problems = len(shuffled_problems)
        val_size = int(total_problems * self.validation_split)

        # Ensure at least one problem in each set if possible
        if val_size == 0 and total_problems > 1:
            val_size = 1
        elif val_size == total_problems and total_problems > 1:
            val_size = total_problems - 1

        # Split data
        validation_problems = shuffled_problems[:val_size]
        training_problems = shuffled_problems[val_size:]

        return training_problems, validation_problems

    def _problem_to_dict(self, problem: TrainingProblem) -> dict[str, Any]:
        """
        Convert TrainingProblem to dictionary format for training.

        Args:
            problem: TrainingProblem object

        Returns:
            Problem dictionary
        """
        return problem.dict()

    async def generate_synthetic_problems(self, count: int = 10) -> list[dict[str, Any]]:
        """
        Generate synthetic coding problems for training diversity.

        Args:
            count: Number of synthetic problems to generate

        Returns:
            List of synthetic problem dictionaries
        """
        synthetic_problems = []

        # Problem templates for generation
        templates = [
            {
                "template": "Write a function to reverse a {data_type}",
                "solutions": {
                    "string": "def reverse_string(s: str) -> str:\n    return s[::-1]",
                    "list": "def reverse_list(lst: list) -> list:\n    return lst[::-1]",
                },
                "test_cases": {
                    "string": [
                        {"input": "hello", "expected_output": "olleh"},
                        {"input": "world", "expected_output": "dlrow"},
                        {"input": "", "expected_output": ""},
                    ],
                    "list": [
                        {"input": [1, 2, 3], "expected_output": [3, 2, 1]},
                        {"input": [], "expected_output": []},
                        {"input": [5], "expected_output": [5]},
                    ],
                },
            },
            {
                "template": "Write a function to find the {operation} of two numbers",
                "solutions": {
                    "sum": "def add_numbers(a: int, b: int) -> int:\n    return a + b",
                    "maximum": "def find_max(a: int, b: int) -> int:\n    return max(a, b)",
                    "minimum": "def find_min(a: int, b: int) -> int:\n    return min(a, b)",
                },
                "test_cases": {
                    "sum": [
                        {"input": [2, 3], "expected_output": 5},
                        {"input": [0, 0], "expected_output": 0},
                        {"input": [-1, 1], "expected_output": 0},
                    ],
                    "maximum": [
                        {"input": [2, 3], "expected_output": 3},
                        {"input": [5, 5], "expected_output": 5},
                        {"input": [-1, -2], "expected_output": -1},
                    ],
                    "minimum": [
                        {"input": [2, 3], "expected_output": 2},
                        {"input": [5, 5], "expected_output": 5},
                        {"input": [-1, -2], "expected_output": -2},
                    ],
                },
            },
        ]

        for i in range(count):
            template = random.choice(templates)

            if template["template"] == "Write a function to reverse a {data_type}":
                data_type = random.choice(["string", "list"])
                problem = template["template"].format(data_type=data_type)
                solution = template["solutions"][data_type]
                test_cases = template["test_cases"][data_type]
            else:
                operation = random.choice(["sum", "maximum", "minimum"])
                problem = template["template"].format(operation=operation)
                solution = template["solutions"][operation]
                test_cases = template["test_cases"][operation]

            synthetic_problem = {
                "id": f"synthetic_{i:04d}",
                "problem": problem,
                "difficulty": random.choice(["easy", "medium"]),
                "test_cases": test_cases,
                "solution": solution,
                "tags": ["synthetic", "basic"],
                "language": "python",
            }

            synthetic_problems.append(synthetic_problem)

        self.logger.info(f"Generated {count} synthetic problems")
        return synthetic_problems

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about loaded training data.

        Returns:
            Statistics dictionary
        """
        if self._problems is None:
            return {"status": "not_loaded"}

        # Count by difficulty
        difficulty_counts = {}
        tag_counts = {}
        language_counts = {}

        for problem in self._problems:
            # Difficulty
            diff = problem.difficulty
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

            # Tags
            for tag in problem.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Language
            lang = problem.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

        return {
            "total_problems": len(self._problems),
            "training_problems": len(self._train_data) if self._train_data else 0,
            "validation_problems": len(self._validation_data) if self._validation_data else 0,
            "validation_split": self.validation_split,
            "difficulty_distribution": difficulty_counts,
            "tag_distribution": tag_counts,
            "language_distribution": language_counts,
        }
