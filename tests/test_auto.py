import json
import tempfile
import unittest
from pathlib import Path

from promptos.auto import AutoOptimizationPolicy, AutoPromptOptimizer
from promptos.core import Judgment, ModelResponse, Sample, SignalSpec, SoftSignal, TaskSpec
from promptos.layered import JsonOutputValidator, LayeredEvaluationPolicy


class AutoTaskModel:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, inputs):
        self.calls += 1
        label = "B" if "better" in prompt else "A"
        return ModelResponse(
            json.dumps({"label": label, "confidence": 0.9}),
            "fake-task",
        )


class AutoJudge:
    def __init__(self):
        self.calls = 0

    def score(self, signal_spec, sample, output, context=None):
        self.calls += 1
        score = 0.9 if '"B"' in output else 0.5
        return Judgment(score, {"quality": score}, [], "B is preferred", 0.9)


class RoundAwareGenerator:
    def __init__(self):
        self.calls = 0
        self.feedback = []

    def propose(self, prompt, task, signal_spec, feedback):
        self.calls += 1
        self.feedback.append(feedback)
        return ["better prompt"] if "better" not in prompt else ["worse prompt"]


def approved_signals():
    return SignalSpec(
        "auto-test",
        1,
        "Return the preferred classification.",
        [],
        [SoftSignal("quality", "Classification quality", 1.0)],
        "approved",
    )


def task_spec():
    return TaskSpec(
        "classification",
        ["id"],
        "Return JSON.",
        {
            "type": "object",
            "required": ["label", "confidence"],
            "properties": {
                "label": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
    )


class AutoOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_unattended_rounds_feedback_memory_stop_and_final_review(self):
        samples = [
            Sample(
                "boundary",
                {"id": "boundary"},
                metadata={
                    "sample_kind": "boundary_probe",
                    "boundary_note": "ambiguous",
                },
            ),
            Sample("ordinary", {"id": "ordinary"}),
        ]
        model, judge, generator = AutoTaskModel(), AutoJudge(), RoundAwareGenerator()
        optimizer = AutoPromptOptimizer(
            model,
            judge,
            generator,
            JsonOutputValidator(task_spec().output_schema),
            LayeredEvaluationPolicy(
                max_candidates=1,
                dynamic_top_k=1,
                judge_max_cases_per_prompt=2,
                human_review_top_k=2,
                task_model_max_calls=30,
                judge_model_max_calls=10,
            ),
            AutoOptimizationPolicy(
                max_rounds=4,
                stop_after_no_improvement=2,
                minimum_improvement=0.01,
                retain_risk_memory=True,
            ),
        )
        result = optimizer.optimize(
            "initial prompt",
            task_spec(),
            approved_signals(),
            samples,
            self.root,
            run_id="auto-resume",
        )

        self.assertEqual(result.champion_prompt, "better prompt")
        self.assertEqual(result.rounds_completed, 2)
        self.assertEqual(
            result.stop_reason,
            "hard_failures_zero_and_no_improvement",
        )
        self.assertIn("boundary", result.risk_memory_case_ids)
        self.assertTrue((result.run_dir / "human_review_top20.csv").exists())
        self.assertTrue((result.run_dir / "champion_prompt.md").exists())
        for index in range(2):
            round_dir = result.run_dir / f"round_{index:02d}"
            self.assertTrue((round_dir / "failure_summary.json").exists())
            self.assertTrue((round_dir / "round_decision.json").exists())
            self.assertFalse((round_dir / "human_review_top20.csv").exists())
        self.assertIn("lowest_judge_results", generator.feedback[1])

        resumed_model = AutoTaskModel()
        resumed_judge = AutoJudge()
        resumed_generator = RoundAwareGenerator()
        resumed = AutoPromptOptimizer(
            resumed_model,
            resumed_judge,
            resumed_generator,
            JsonOutputValidator(task_spec().output_schema),
            optimizer.layered_policy,
            optimizer.auto_policy,
        ).optimize(
            "initial prompt",
            task_spec(),
            approved_signals(),
            samples,
            self.root,
            run_id="auto-resume",
        )
        self.assertEqual(resumed.champion_prompt, "better prompt")
        self.assertEqual(resumed_model.calls, 0)
        self.assertEqual(resumed_judge.calls, 0)
        self.assertEqual(resumed_generator.calls, 0)


if __name__ == "__main__":
    unittest.main()
