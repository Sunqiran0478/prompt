import csv
import json
import tempfile
import unittest
from pathlib import Path

from promptos.core import (
    Judgment,
    ModelResponse,
    Sample,
    SignalSpec,
    SoftSignal,
    TaskSpec,
)
from promptos.layered import (
    JsonOutputValidator,
    LayeredEvaluationPolicy,
    LayeredPromptOptimizer,
)
from promptos.plugins.finance_classification import (
    FinanceOutputValidator,
    FinanceTaxonomy,
)


class FakeTaskModel:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, inputs):
        self.calls += 1
        sample_id = inputs["id"]
        if sample_id == "hard":
            text = "not-json"
        else:
            confidence = 0.4 if sample_id.startswith("risk") else 0.95
            label = "B" if "candidate" in prompt and sample_id == "boundary" else "A"
            text = json.dumps({"label": label, "confidence": confidence})
        return ModelResponse(text, "fake-task", 2, 1)


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def propose(self, prompt, task, signal_spec, feedback):
        self.calls += 1
        return ["candidate prompt"]


class FakeJudge:
    def __init__(self):
        self.calls = []

    def score(self, signal_spec, sample, output, context=None):
        self.calls.append((sample.id, output, context))
        score = 0.9 if '"label": "B"' in output else 0.6
        return Judgment(score, {"quality": score}, [], "fake rationale", 0.8)


def approved_signals():
    return SignalSpec(
        "test", 1, "Return a valid classification.", [],
        [SoftSignal("quality", "Output quality", 1.0)], "approved",
    )


def task_spec():
    return TaskSpec(
        "classification", ["id"], "Return JSON.",
        {
            "type": "object",
            "required": ["label", "confidence"],
            "properties": {
                "label": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
    )


class LayeredEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_layered_funnel_shared_queue_hard_fail_and_resume(self):
        samples = [
            Sample("boundary", {"id": "boundary"}, metadata={
                "sample_kind": "boundary_probe",
                "boundary_note": "ambiguous wording",
            }),
            Sample("risk-1", {"id": "risk-1"}),
            Sample("safe", {"id": "safe"}),
            Sample("hard", {"id": "hard"}),
        ]
        policy = LayeredEvaluationPolicy(
            max_candidates=1,
            dynamic_top_k=1,
            judge_max_cases_per_prompt=2,
            human_review_top_k=2,
            task_model_max_calls=20,
            judge_model_max_calls=4,
        )
        model, judge, generator = FakeTaskModel(), FakeJudge(), FakeGenerator()
        optimizer = LayeredPromptOptimizer(
            model, judge, generator, JsonOutputValidator(task_spec().output_schema), policy,
        )
        result = optimizer.optimize(
            "baseline", task_spec(), approved_signals(), samples, self.tmp_path, run_id="resume-me",
        )

        self.assertEqual(result.judge_case_ids, ["boundary", "risk-1"])
        self.assertEqual(result.champion_prompt, "candidate prompt")
        self.assertEqual(len(judge.calls), 4)
        self.assertEqual({call[0] for call in judge.calls}, {"boundary", "risk-1"})
        boundary_context = next(call[2] for call in judge.calls if call[0] == "boundary")
        self.assertEqual(boundary_context["boundary_note"], "ambiguous wording")
        self.assertEqual(set(boundary_context["comparison_outputs"]), {"0", "1"})
        self.assertTrue(all(call[0] != "hard" for call in judge.calls))
        run_dir = result.run_dir
        for name in (
        "task_outputs.jsonl",
        "deterministic_checks.jsonl",
        "risk_scores.jsonl",
        "judge_queue.json",
        "judge_results.jsonl",
        "prompt_comparison.json",
        "human_review_top20.csv",
        "run.json",
        ):
            self.assertTrue((run_dir / name).exists())
        risk_rows = [
        json.loads(line)
        for line in (run_dir / "risk_scores.jsonl").read_text().splitlines()
        ]
        hard = next(row for row in risk_rows if row["sample_id"] == "hard")
        self.assertTrue(hard["hard_failure"])
        self.assertTrue(hard["risk_signals"][0]["reason"])
        with (run_dir / "human_review_top20.csv").open(newline="") as file:
            self.assertLessEqual(len(list(csv.DictReader(file))), 2)

        resumed_model, resumed_judge, resumed_generator = FakeTaskModel(), FakeJudge(), FakeGenerator()
        resumed = LayeredPromptOptimizer(
            resumed_model,
            resumed_judge,
            resumed_generator,
            JsonOutputValidator(task_spec().output_schema),
            policy,
        ).optimize("baseline", task_spec(), approved_signals(), samples, self.tmp_path, run_id="resume-me")
        self.assertEqual(resumed.champion_prompt, result.champion_prompt)
        self.assertEqual(resumed_model.calls, 0)
        self.assertEqual(resumed_judge.calls, [])
        self.assertEqual(resumed_generator.calls, 0)


    def test_fixed_set_over_cap_fails_before_model_call(self):
        samples = [
        Sample(str(index), {"id": str(index)}, metadata={"sample_kind": "boundary_probe"})
        for index in range(2)
        ]
        model = FakeTaskModel()
        optimizer = LayeredPromptOptimizer(
        model,
        FakeJudge(),
        FakeGenerator(),
        JsonOutputValidator(task_spec().output_schema),
        LayeredEvaluationPolicy(max_candidates=0, judge_max_cases_per_prompt=1),
        )
        with self.assertRaisesRegex(ValueError, "exceeding cap"):
            optimizer.optimize("baseline", task_spec(), approved_signals(), samples, self.tmp_path)
        self.assertEqual(model.calls, 0)


    def test_finance_validator_rejects_invalid_taxonomy_and_schema(self):
        taxonomy = FinanceTaxonomy(
        "1", {"L3-1": "L2-1"}, {"L2-1": "Payments", "L2-2": "Loans", "L3-1": "Card"}, "",
        )
        validator = FinanceOutputValidator(taxonomy)
        sample = Sample("1", {"query": "test"})

        valid = '{"L2_id":"L2-1","L3_id":"L3-1","confidence":0.8,"reason":"x"}'
        self.assertEqual(validator.validate(sample, valid)[1], [])
        invalid_parent = '{"L2_id":"L2-2","L3_id":"L3-1","confidence":0.8,"reason":"x"}'
        self.assertEqual(
            {item.code for item in validator.validate(sample, invalid_parent)[1]},
            {"valid_taxonomy"},
        )
        missing = '{"L2_id":"L2-1","L3_id":"L3-1","confidence":0.8}'
        self.assertEqual(
            {item.code for item in validator.validate(sample, missing)[1]},
            {"valid_schema"},
        )
        confidence = '{"L2_id":"L2-1","L3_id":"L3-1","confidence":2,"reason":"x"}'
        self.assertEqual(
            {item.code for item in validator.validate(sample, confidence)[1]},
            {"valid_schema"},
        )


    def test_finance_shape_200_selects_50_fixed_plus_30_dynamic(self):
        samples = [
        Sample(
            f"boundary-{index:03d}",
            {"id": f"boundary-{index:03d}"},
            metadata={"sample_kind": "boundary_probe", "boundary_note": "probe"},
        )
        for index in range(50)
        ]
        samples.extend(
        Sample(
            f"risk-{index:03d}" if index < 40 else f"safe-{index:03d}",
            {"id": f"risk-{index:03d}" if index < 40 else f"safe-{index:03d}"},
        )
        for index in range(150)
        )
        model, judge = FakeTaskModel(), FakeJudge()
        policy = LayeredEvaluationPolicy(
        max_candidates=0,
        dynamic_top_k=30,
        judge_max_cases_per_prompt=80,
        human_review_top_k=20,
        task_model_max_calls=300,
        judge_model_max_calls=80,
        )
        result = LayeredPromptOptimizer(
        model, judge, FakeGenerator(), JsonOutputValidator(task_spec().output_schema), policy,
        ).optimize("baseline", task_spec(), approved_signals(), samples, self.tmp_path)

        self.assertEqual(len(result.judge_case_ids), 80)
        self.assertEqual(
            sum(sample_id.startswith("boundary-") for sample_id in result.judge_case_ids),
            50,
        )
        self.assertEqual(
            result.judge_case_ids[50:],
            [f"risk-{index:03d}" for index in range(30)],
        )
        self.assertEqual(len(judge.calls), 80)
        self.assertEqual(result.champion_status, "provisional_silver_or_unlabeled")
        with (result.run_dir / "human_review_top20.csv").open(newline="") as file:
            self.assertEqual(len(list(csv.DictReader(file))), 20)


if __name__ == "__main__":
    unittest.main()
