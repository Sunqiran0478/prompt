import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def compare(self, signal_spec, sample, outputs, context=None):
        self.calls += 1
        return {
            index: Judgment(
                0.9 if '"B"' in output else 0.5,
                {"quality": 0.9 if '"B"' in output else 0.5},
                [],
                "B is preferred",
                0.9,
            )
            for index, output in outputs.items()
        }


class RoundAwareGenerator:
    def __init__(self):
        self.calls = 0
        self.feedback = []
        self.model = SimpleNamespace(last_response=None)

    def propose(self, prompt, task, signal_spec, feedback):
        self.calls += 1
        self.feedback.append(feedback)
        self.model.last_response = ModelResponse(
            '{"candidates": []}',
            "fake-generator",
            input_tokens=7,
            output_tokens=3,
            cost_usd=0.01,
        )
        return ["better prompt"] if "better" not in prompt else ["worse prompt"]


class FailSecondGenerator(RoundAwareGenerator):
    def propose(self, prompt, task, signal_spec, feedback):
        if self.calls == 1:
            raise RuntimeError("planned interruption")
        return super().propose(prompt, task, signal_spec, feedback)


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
                {"id": "boundary,\n中文"},
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
                dynamic_top_k=0,
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
            task_config={
                "version": 1,
                "models": {
                    "task_model": "fake-task",
                    "judge_model": "fake-judge",
                    "api_key": "must-not-appear-in-report",
                },
            },
        )

        self.assertEqual(result.champion_prompt, "better prompt")
        self.assertEqual(result.rounds_completed, 3)
        self.assertEqual(
            result.stop_reason,
            "no_material_improvement",
        )
        self.assertIn("boundary", result.risk_memory_case_ids)
        self.assertTrue((result.run_dir / "human_review_top20.csv").exists())
        self.assertTrue((result.run_dir / "champion_prompt.md").exists())
        self.assertTrue((result.run_dir / "experiment_report.md").exists())
        report = (result.run_dir / "experiment_report.md").read_text(encoding="utf-8")
        self.assertIn("N/A (n=0)", report)
        self.assertIn("unavailable", report)
        self.assertNotIn("must-not-appear-in-report", report)
        self.assertIn("[REDACTED]", report)
        manifest = json.loads((result.run_dir / "report_manifest.json").read_text())
        self.assertNotIn("must-not-appear-in-report", json.dumps(manifest))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["model_calls_added"], 0)
        for index in range(3):
            round_dir = result.run_dir / f"round_{index:02d}"
            self.assertTrue((round_dir / "failure_summary.json").exists())
            self.assertTrue((round_dir / "round_decision.json").exists())
            self.assertTrue((round_dir / "generator_metadata.json").exists())
            generator_metadata = json.loads(
                (round_dir / "generator_metadata.json").read_text()
            )
            self.assertEqual(generator_metadata["model"], "fake-generator")
            self.assertTrue((round_dir / "risk_case_details.csv").exists())
            self.assertTrue((round_dir / "prompt_evolution.md").exists())
            self.assertFalse((round_dir / "human_review_top20.csv").exists())
            evolution = (round_dir / "prompt_evolution.md").read_text(encoding="utf-8")
            self.assertIn("完整 Prompt", evolution)
            self.assertIn("Change summary", evolution)
        with (result.run_dir / "round_00" / "risk_case_details.csv").open(
            encoding="utf-8",
            newline="",
        ) as file:
            risk_rows = list(csv.DictReader(file))
        boundary_rows = [row for row in risk_rows if row["sample_id"] == "boundary"]
        self.assertTrue(boundary_rows)
        self.assertIn("boundary,\\n中文", boundary_rows[0]["query"])
        self.assertEqual(boundary_rows[0]["judge_model"], "unavailable")
        self.assertIn("fixed_boundary", boundary_rows[0]["selection_sources"])
        with (result.run_dir / "round_01" / "risk_case_details.csv").open(
            encoding="utf-8",
            newline="",
        ) as file:
            resumed_risk_rows = list(csv.DictReader(file))
        remembered = [row for row in resumed_risk_rows if row["sample_id"] == "boundary"]
        self.assertTrue(any("risk_memory" in row["selection_sources"] for row in remembered))
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
            task_config={
                "version": 1,
                "models": {
                    "task_model": "fake-task",
                    "judge_model": "fake-judge",
                    "api_key": "must-not-appear-in-report",
                },
            },
        )
        self.assertEqual(resumed.champion_prompt, "better prompt")
        self.assertEqual(resumed_model.calls, 0)
        self.assertEqual(resumed_judge.calls, 0)
        self.assertEqual(resumed_generator.calls, 0)

    def test_interrupted_run_keeps_an_incomplete_report(self):
        samples = [
            Sample(
                "boundary",
                {"id": "boundary"},
                metadata={"sample_kind": "boundary_probe", "boundary_note": "edge"},
            ),
        ]
        optimizer = AutoPromptOptimizer(
            AutoTaskModel(),
            AutoJudge(),
            FailSecondGenerator(),
            JsonOutputValidator(task_spec().output_schema),
            LayeredEvaluationPolicy(
                max_candidates=1,
                dynamic_top_k=0,
                judge_max_cases_per_prompt=1,
                task_model_max_calls=20,
                judge_model_max_calls=10,
            ),
            AutoOptimizationPolicy(max_rounds=3, stop_after_no_improvement=2),
        )
        with self.assertRaisesRegex(RuntimeError, "planned interruption"):
            optimizer.optimize(
                "initial prompt",
                task_spec(),
                approved_signals(),
                samples,
                self.root,
                run_id="interrupted",
            )
        run_dir = self.root / "interrupted"
        manifest = json.loads((run_dir / "report_manifest.json").read_text())
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(manifest["completed_rounds"], ["round_00"])
        self.assertFalse((run_dir / "run.json").exists())

    def test_first_round_hard_failure_match_stops_immediately(self):
        samples = [
            Sample(
                "boundary",
                {"id": "boundary"},
                metadata={"sample_kind": "boundary_probe"},
            ),
        ]
        optimizer = AutoPromptOptimizer(
            AutoTaskModel(),
            AutoJudge(),
            RoundAwareGenerator(),
            JsonOutputValidator(task_spec().output_schema),
            LayeredEvaluationPolicy(
                max_candidates=1,
                dynamic_top_k=0,
                judge_max_cases_per_prompt=1,
                task_model_max_calls=20,
                judge_model_max_calls=10,
            ),
            AutoOptimizationPolicy(
                max_rounds=4,
                stop_after_no_improvement=2,
                first_round_hard_failure_stop_count=0,
            ),
        )
        result = optimizer.optimize(
            "initial prompt",
            task_spec(),
            approved_signals(),
            samples,
            self.root,
            run_id="first-round-stop",
        )
        self.assertEqual(result.rounds_completed, 1)
        self.assertEqual(
            result.stop_reason,
            "first_round_hard_failure_count_matched_stop_condition",
        )


if __name__ == "__main__":
    unittest.main()
