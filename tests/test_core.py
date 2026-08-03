import json
import csv
import tempfile
import unittest
from pathlib import Path

from promptos.adapters import LLMPromptGenerator, LLMSignalCompiler, ReferenceJudge, RuleBasedDemoModel, TemplatePromptGenerator
from promptos.core import Budget, ModelResponse, PromptOptimizer, RunStore, Sample, SignalEvaluator, SignalSpec, SoftSignal, TaskSpec
from promptos.provenance import Annotation, apply_annotations, export_review_csv, import_review_csv, select_review_cases
from promptos.datasets import split_gold_samples, write_split
from promptos.config import load_task_config
from promptos.plugins.finance_classification import FinanceTaxonomy, default_prompt, signal_spec, task_spec, validate_output


class PromptOSTests(unittest.TestCase):
    def setUp(self):
        self.samples = [Sample("a", {"text": "hello"}, "HELLO", "gold_human")]
        self.signals = SignalSpec("quality", 1, "Return exact uppercase text.", [],
                                  [SoftSignal("exactness", "Matches the reference.", 1.0)], "approved")
        self.task = TaskSpec("arbitrary-text-transform", ["text"], "Return uppercase text.")

    def test_any_input_field_and_approved_signal_spec_optimize(self):
        optimizer = PromptOptimizer(RuleBasedDemoModel(), SignalEvaluator(ReferenceJudge()), TemplatePromptGenerator())
        result = optimizer.optimize("Transform input", self.task, self.signals, self.samples, Budget(max_calls=10), rounds=2)
        self.assertEqual(result.champion_score, 1.0)
        self.assertIn("uppercase", result.champion_prompt.lower())

    def test_draft_signal_spec_cannot_start_run(self):
        draft = SignalSpec("quality", 1, "criterion", [], [SoftSignal("quality", "good", 1.0)])
        optimizer = PromptOptimizer(RuleBasedDemoModel(), SignalEvaluator(ReferenceJudge()), TemplatePromptGenerator())
        with self.assertRaisesRegex(ValueError, "approved"):
            optimizer.optimize("prompt", self.task, draft, self.samples, Budget(), rounds=0)

    def test_llm_signal_compiler_recovers_from_all_zero_weights(self):
        class ZeroWeightModel:
            model = "test-model"

            def complete_json(self, system, user):
                return {
                    "hard_constraints": [],
                    "soft_signals": [
                        {"name": "correctness", "criterion": "Correct result", "weight": 0},
                        {"name": "clarity", "criterion": "Clear result", "weight": 0},
                    ],
                }

        spec = LLMSignalCompiler(ZeroWeightModel()).compile("Return the correct result.")
        self.assertEqual([signal.weight for signal in spec.soft_signals], [0.5, 0.5])
        spec.validate()

    def test_llm_prompt_generator_accepts_fenced_array_response(self):
        class ArrayResponseModel:
            def _chat(self, system, user):
                return ModelResponse(
                    '```json\n[{"current_prompt":"candidate one"},'
                    '{"candidates":["candidate two"]}]\n```',
                    "fake",
                )

        candidates = list(
            LLMPromptGenerator(ArrayResponseModel()).propose(
                "initial", self.task, self.signals, "improve boundaries",
            )
        )
        self.assertEqual(candidates, ["candidate one", "candidate two"])

    def test_llm_prompt_generator_falls_back_to_auditable_candidates(self):
        class EchoModel:
            def _chat(self, system, user):
                return ModelResponse(
                    '```json\n[{"current_prompt":"initial"}]\n```',
                    "fake",
                )

        candidates = list(
            LLMPromptGenerator(EchoModel()).propose(
                "initial", self.task, self.signals, "improve boundaries",
            )
        )
        distinct = [value for value in candidates if value != "initial"]
        self.assertEqual(len(distinct), 2)
        self.assertIn("路由冲突", distinct[0])
        self.assertIn("歧义与输出自检", distinct[1])

    def test_run_store_records_provenance_without_secrets(self):
        optimizer = PromptOptimizer(RuleBasedDemoModel(), SignalEvaluator(ReferenceJudge()), TemplatePromptGenerator())
        result = optimizer.optimize("uppercase", self.task, self.signals, self.samples, Budget(max_calls=5), rounds=0)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = RunStore(Path(directory)).save(result, self.task, self.signals, self.samples, {"version": 1})
            run = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(run["label_sources"]["gold_human"], 1)
            self.assertNotIn("api_key", json.dumps(run).lower())
            self.assertEqual(run["task_config"]["version"], 1)
            evaluation = json.loads((run_dir / "evaluations.jsonl").read_text().splitlines()[0])
            self.assertEqual(evaluation["results"][0]["model"], "local-demo")

    def test_annotation_guardrails_and_unlabeled_review_selection(self):
        unlabeled = Sample("review", {"answer": "claim"}, metadata={"uncertainty": 0.8})
        queue = select_review_cases([unlabeled], 5)
        self.assertEqual(queue[0]["reason"], "uncertainty")
        with self.assertRaises(ValueError):
            apply_annotations([unlabeled], [Annotation("review", True, "gold_human")])
        labeled = apply_annotations([unlabeled], [Annotation("review", True, "gold_human", reviewer="alice")])
        self.assertEqual(labeled[0].label_source, "gold_human")

    def test_gold_split_is_deterministic_and_writes_lock_manifest(self):
        samples = [Sample(str(index), {"x": index}, index, "gold_human") for index in range(20)]
        first = split_gold_samples(samples, "seed")
        self.assertEqual(first, split_gold_samples(samples, "seed"))
        with tempfile.TemporaryDirectory() as directory:
            root = write_split(Path(directory) / "locked", first, "seed")
            manifest = json.loads((root / "split_manifest.json").read_text())
            self.assertTrue(manifest["locked"])
            self.assertTrue((root / "final_test.jsonl").exists())

    def test_task_config_resolves_paths_and_rejects_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "task.json"
            config_path.write_text(json.dumps({"version": 1, "task": {"name": "fact-check", "input_fields": ["answer"], "output_description": "Return a verdict."},
                                                "dataset": {"path": "data.jsonl", "expected_field": "verdict"}, "signals": {"path": "signals.json"},
                                                "models": {"task_model": "example", "task_response_format": "json_object",
                                                           "task_max_tokens": 2048},
                                                "optimization": {"initial_prompt": "Check facts."}}))
            config = load_task_config(config_path)
            self.assertEqual(config.dataset.path, root / "data.jsonl")
            self.assertEqual(config.signal_spec, root / "signals.json")
            self.assertEqual(config.models.task_response_format, "json_object")
            self.assertEqual(config.models.task_max_tokens, 2048)
            config_path.write_text(config_path.read_text().replace('"task_model": "example"', '"api_key": "forbidden"'))
            with self.assertRaisesRegex(ValueError, "credentials"):
                load_task_config(config_path)

    def test_layered_finance_config_resolves_plugin_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "task.json"
            config_path.write_text(json.dumps({
                "version": 1,
                "task": {
                    "name": "finance",
                    "input_fields": ["query"],
                    "output_description": "Classify.",
                },
                "dataset": {"path": "sample.jsonl"},
                "signals": {"path": "signals.json"},
                "models": {"task_model": "deepseek-chat"},
                "plugin": {
                    "name": "finance_classification",
                    "taxonomy_path": "taxonomy.json",
                },
                "evaluation": {
                    "mode": "layered",
                    "fixed_sample_kinds": ["boundary_probe"],
                    "dynamic_top_k": 30,
                    "judge_max_cases_per_prompt": 80,
                },
            }))
            config = load_task_config(config_path)
            self.assertEqual(config.evaluation.mode, "layered")
            self.assertEqual(config.evaluation.dynamic_top_k, 30)
            self.assertEqual(config.plugin.taxonomy_path, root / "taxonomy.json")
            self.assertEqual(config.optimization.initial_prompt, "")

    def test_human_review_csv_only_promotes_explicit_human_decisions(self):
        samples = [Sample("one", {"answer": "claim"})]
        queue = [{"sample_id": "one"}]
        silver = [Annotation("one", {"verdict": False}, "silver_auto", judge_run_id="judge-1")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            export_review_csv(samples, queue, silver, path)
            with path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            rows[0].update({"decision": "edit", "reviewed_value": '{"verdict": true}', "reviewer": "reviewer@example", "review_rationale": "Checked source."})
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            annotations, rejected = import_review_csv(path)
            self.assertFalse(rejected)
            self.assertEqual(annotations[0].source, "gold_human")
            self.assertEqual(annotations[0].value, {"verdict": True})

    def test_finance_plugin_keeps_taxonomy_logic_outside_core(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taxonomy.json"
            path.write_text(json.dumps({"version": "test", "L2_list": [{"id": "L2-1", "name": "Research", "L3_list": [
                {"id": "L3-1-1", "name": "Company", "definition": "Research a company."}
            ]}]}))
            taxonomy = FinanceTaxonomy.load(path)
            self.assertIn("L3-1-1", default_prompt(taxonomy))
            self.assertEqual(task_spec().input_fields, ["query"])
            self.assertEqual(validate_output({"L2_id": "L2-1", "L3_id": "L3-1-1", "confidence": 0.8, "reason": "x"}, taxonomy), [])
            self.assertEqual(validate_output({"L2_id": "L2-9", "L3_id": "L3-1-1", "confidence": 0.8, "reason": "x"}, taxonomy), ["valid_taxonomy"])
            self.assertEqual(signal_spec(taxonomy).status, "draft")


if __name__ == "__main__":
    unittest.main()
