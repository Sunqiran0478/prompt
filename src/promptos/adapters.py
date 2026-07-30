"""Local reference adapters and an OpenAI-compatible HTTP adapter."""
from __future__ import annotations

import json
import os
import hashlib
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any, Iterable

from .core import HardConstraint, Judgment, ModelResponse, PromptGenerator, Sample, SignalSpec, SoftSignal, TaskSpec


class OpenAICompatibleModel:
    """Minimal dependency-free Chat Completions adapter.

    API keys are read from the environment at runtime and are never persisted.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.0,
                 max_retries: int = 3, cache_dir: str | None = None, input_token_usd: float = 0.0, output_token_usd: float = 0.0):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key_env, self.temperature = api_key_env, temperature
        self.max_retries = max_retries
        self.cache_dir = cache_dir or os.environ.get("PROMPTOS_CACHE_DIR")
        self.input_token_usd, self.output_token_usd = input_token_usd, output_token_usd

    def _cache_path(self, body: bytes) -> str | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(f"{self.base_url}:{body.decode()}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.json")

    def _chat(self, system: str, user: str) -> ModelResponse:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing {self.api_key_env}; provide it only as an environment variable.")
        body = json.dumps({"model": self.model, "temperature": self.temperature,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        cache_path = self._cache_path(body)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as file:
                payload = json.load(file)
            payload["_promptos_cached"] = True
        else:
            request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            for attempt in range(self.max_retries + 1):
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        payload = json.loads(response.read())
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
                    status = getattr(error, "code", 0)
                    if attempt >= self.max_retries or (status and status < 429 and status < 500):
                        raise RuntimeError(f"Chat request failed after {attempt + 1} attempt(s): {error}") from error
                    time.sleep(min(8.0, 0.5 * (2 ** attempt)))
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as file:
                    json.dump(payload, file, ensure_ascii=False)
        usage = payload.get("usage", {})
        input_tokens, output_tokens = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        response = ModelResponse(payload["choices"][0]["message"]["content"], self.model,
            input_tokens, output_tokens, input_tokens * self.input_token_usd + output_tokens * self.output_token_usd, payload)
        self.last_response = response
        return response

    def generate(self, prompt: str, inputs: dict[str, Any]) -> ModelResponse:
        return self._chat(prompt, json.dumps(inputs, ensure_ascii=False, indent=2))

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        text = self._chat(system, user).text
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Model did not return a JSON object.")
        return json.loads(text[start:end + 1])


class LLMJudge:
    def __init__(self, model: OpenAICompatibleModel):
        self.model = model

    def score(
        self,
        signal_spec: SignalSpec,
        sample: Sample,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> Judgment:
        rubric = {"hard_constraints": [asdict(item) for item in signal_spec.hard_constraints],
                  "soft_signals": [asdict(item) for item in signal_spec.soft_signals]}
        payload = self.model.complete_json(
            "You are a strict evaluation judge. Do not improve the answer. Return JSON only.",
            json.dumps({"acceptance_criteria": signal_spec.acceptance_criteria, "rubric": rubric,
                        "inputs": sample.inputs, "reference": sample.expected, "candidate_output": output,
                        "evaluation_context": context or {},
                        "required_output": {"hard_failures": ["constraint name"], "signal_scores": {"signal name": 0.0},
                                            "score": 0.0, "confidence": 0.0, "rationale": "brief reason"}}, ensure_ascii=False))
        allowed = {signal.name for signal in signal_spec.soft_signals}
        scores = {str(key): max(0.0, min(1.0, float(value))) for key, value in payload.get("signal_scores", {}).items()
                  if str(key) in allowed}
        weights = {signal.name: signal.weight for signal in signal_spec.soft_signals}
        weighted = sum(scores.get(name, 0.0) * weight for name, weight in weights.items()) / sum(weights.values())
        declared_hard = {item.name for item in signal_spec.hard_constraints}
        hard = [str(item) for item in payload.get("hard_failures", []) if str(item) in declared_hard]
        response = getattr(self.model, "last_response", None)
        if response:
            payload = {**payload, "_promptos_judge_model": response.model,
                       "_promptos_judge_tokens": response.input_tokens + response.output_tokens,
                       "_promptos_judge_cost_usd": response.cost_usd}
        return Judgment(0.0 if hard else max(0.0, min(1.0, weighted)), scores, hard,
                        str(payload.get("rationale", "")), float(payload.get("confidence", 0.0)), payload)


class LLMSilverAnnotator:
    """Independent judge-generated labels. Results remain silver until reviewed."""
    def __init__(self, model: OpenAICompatibleModel):
        self.model = model

    def annotate(self, signal_spec: SignalSpec, sample: Sample) -> tuple[Any, str]:
        payload = self.model.complete_json(
            "You independently label task examples for later human review. Return JSON only.",
            json.dumps({"acceptance_criteria": signal_spec.acceptance_criteria, "inputs": sample.inputs,
                        "required_output": {"value": "the task's best expected output; JSON values allowed", "rationale": "brief evidence-based explanation"}}, ensure_ascii=False))
        return payload.get("value"), str(payload.get("rationale", ""))


class LLMSignalCompiler:
    def __init__(self, model: OpenAICompatibleModel | None = None):
        self.model = model

    def compile(self, acceptance_criteria: str, spec_id: str = "signals", version: int = 1) -> SignalSpec:
        if self.model is None:
            signals = [SoftSignal("acceptance_alignment", acceptance_criteria, 0.7),
                       SoftSignal("output_clarity", "Is the output clear, specific, and useful?", 0.3)]
            return SignalSpec(spec_id, version, acceptance_criteria, [], signals, "draft", "local_template")
        payload = self.model.complete_json(
            "Convert product acceptance criteria into an auditable evaluation rubric. Return JSON only. "
            "Every soft-signal weight must be positive, and all weights should sum to 1.0.",
            json.dumps({"acceptance_criteria": acceptance_criteria,
                        "required_output": {"hard_constraints": [{"name": "", "description": ""}],
                                            "soft_signals": [{"name": "", "criterion": "", "weight": 1.0}]}}))
        hard = [item for item in payload.get("hard_constraints", []) if item.get("name")]
        soft = [item for item in payload.get("soft_signals", []) if item.get("name")]
        if soft and sum(max(0.0, float(item.get("weight", 0.0))) for item in soft) <= 0:
            equal_weight = 1.0 / len(soft)
            soft = [{**item, "weight": equal_weight} for item in soft]
        return SignalSpec(spec_id, version, acceptance_criteria,
            [HardConstraint(**item) for item in hard],
            [SoftSignal(str(item["name"]), str(item["criterion"]), float(item["weight"])) for item in soft],
            "draft", f"llm:{self.model.model}")


class TemplatePromptGenerator(PromptGenerator):
    def propose(self, prompt: str, task: TaskSpec, signal_spec: SignalSpec, feedback: str) -> Iterable[str]:
        constraints = "\n".join(f"- {item.description}" for item in signal_spec.hard_constraints)
        yield f"{prompt.strip()}\n\nAcceptance criteria:\n{signal_spec.acceptance_criteria}"
        yield f"{prompt.strip()}\n\nFollow these non-negotiable constraints:\n{constraints}" if constraints else f"{prompt.strip()}\n\nBe explicit, structured, and verify requirements before responding."
        yield f"Task: {task.output_description}\n\n{prompt.strip()}\n\nReturn only the requested result and explain uncertainty when evidence is insufficient."


class LLMPromptGenerator(PromptGenerator):
    """Generates prompt variants without exposing evaluation outputs to the task model."""
    def __init__(self, model: OpenAICompatibleModel):
        self.model = model

    def propose(self, prompt: str, task: TaskSpec, signal_spec: SignalSpec, feedback: str) -> Iterable[str]:
        payload = self.model.complete_json(
            "You improve prompts for a task. Return JSON only; do not evaluate any examples.",
            json.dumps({"current_prompt": prompt, "task": asdict(task), "signal_spec": asdict(signal_spec),
                        "feedback": feedback, "required_output": {"candidates": ["prompt text"]}}, ensure_ascii=False))
        return [str(item) for item in payload.get("candidates", [])]


class ReferenceJudge:
    """Offline judge for examples/tests. Exact expected-output match earns one."""
    def score(
        self,
        signal_spec: SignalSpec,
        sample: Sample,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> Judgment:
        del context
        correct = sample.expected is not None and str(sample.expected).strip() == output.strip()
        scores = {signal.name: float(correct) for signal in signal_spec.soft_signals}
        return Judgment(float(correct), scores, [] if correct else ["reference_mismatch"], "Exact reference comparison.", 1.0)


class RuleBasedDemoModel:
    def generate(self, prompt: str, inputs: dict[str, Any]) -> ModelResponse:
        value = str(inputs.get("text", inputs.get("answer", "")))
        return ModelResponse(value.upper() if "uppercase" in prompt.lower() or "大写" in prompt else value, "local-demo")
