"""Local reference adapters and an OpenAI-compatible HTTP adapter."""
from __future__ import annotations

import json
import os
import hashlib
import threading
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

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_retries: int = 3,
        cache_dir: str | None = None,
        input_token_usd: float = 0.0,
        output_token_usd: float = 0.0,
        response_format: str | None = None,
        max_tokens: int | None = None,
    ):
        if response_format not in {None, "text", "json_object"}:
            raise ValueError("response_format must be text or json_object.")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key_env, self.temperature = api_key_env, temperature
        self.max_retries = max_retries
        self.cache_dir = cache_dir or os.environ.get("PROMPTOS_CACHE_DIR")
        self.input_token_usd, self.output_token_usd = input_token_usd, output_token_usd
        self.response_format = response_format
        self.max_tokens = max_tokens
        self._response_local = threading.local()

    @property
    def last_response(self) -> ModelResponse | None:
        return getattr(self._response_local, "value", None)

    @last_response.setter
    def last_response(self, value: ModelResponse | None) -> None:
        self._response_local.value = value

    def _cache_path(self, body: bytes) -> str | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(f"{self.base_url}:{body.decode()}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.json")

    def _chat(self, system: str, user: str) -> ModelResponse:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing {self.api_key_env}; provide it only as an environment variable.")
        request_body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.response_format:
            request_body["response_format"] = {"type": self.response_format}
        if self.max_tokens is not None:
            request_body["max_tokens"] = self.max_tokens
        body = json.dumps(request_body).encode()
        cache_path = self._cache_path(body)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as file:
                payload = json.load(file)
            if self._valid_structured_payload(payload):
                payload["_promptos_cached"] = True
            else:
                payload = None
        else:
            payload = None
        if payload is None:
            request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            for attempt in range(self.max_retries + 1):
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        payload = json.loads(response.read())
                    if not self._valid_structured_payload(payload):
                        raise ValueError(
                            "JSON Output returned empty, invalid, or non-object content."
                        )
                    break
                except (
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    TimeoutError,
                    ValueError,
                ) as error:
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

    def _valid_structured_payload(self, payload: dict[str, Any]) -> bool:
        if self.response_format != "json_object":
            return True
        try:
            content = payload["choices"][0]["message"]["content"]
            value = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return False
        return isinstance(value, dict)

    def generate(self, prompt: str, inputs: dict[str, Any]) -> ModelResponse:
        return self._chat(prompt, json.dumps(inputs, ensure_ascii=False, indent=2))

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        text = self._chat(system, user).text
        if self.response_format == "json_object":
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError("Model did not return a JSON object.")
            return value
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
        response = getattr(self.model, "last_response", None)
        return self._judgment(signal_spec, payload, response)

    def _judgment(
        self,
        signal_spec: SignalSpec,
        payload: dict[str, Any],
        response: ModelResponse | None,
    ) -> Judgment:
        allowed = {signal.name for signal in signal_spec.soft_signals}
        scores = {str(key): max(0.0, min(1.0, float(value))) for key, value in payload.get("signal_scores", {}).items()
                  if str(key) in allowed}
        weights = {signal.name: signal.weight for signal in signal_spec.soft_signals}
        weighted = sum(scores.get(name, 0.0) * weight for name, weight in weights.items()) / sum(weights.values())
        declared_hard = {item.name for item in signal_spec.hard_constraints}
        hard = [str(item) for item in payload.get("hard_failures", []) if str(item) in declared_hard]
        if response:
            payload = {**payload, "_promptos_judge_model": response.model,
                       "_promptos_judge_tokens": response.input_tokens + response.output_tokens,
                       "_promptos_judge_cost_usd": response.cost_usd,
                       "_promptos_judge_cached": bool(response.raw.get("_promptos_cached"))}
        return Judgment(0.0 if hard else max(0.0, min(1.0, weighted)), scores, hard,
                        str(payload.get("rationale", "")), float(payload.get("confidence", 0.0)), payload)

    def compare(
        self,
        signal_spec: SignalSpec,
        sample: Sample,
        outputs: dict[int, str],
        context: dict[str, Any] | None = None,
    ) -> dict[int, Judgment]:
        """Score all Prompt outputs for one case in a single, anonymized Judge call."""
        rubric = {
            "hard_constraints": [asdict(item) for item in signal_spec.hard_constraints],
            "soft_signals": [asdict(item) for item in signal_spec.soft_signals],
        }
        aliases = {index: f"candidate_{index}" for index in sorted(outputs)}
        required = {
            alias: {
                "hard_failures": ["constraint name"],
                "signal_scores": {"signal name": 0.0},
                "confidence": 0.0,
                "rationale": "brief reason",
            }
            for alias in aliases.values()
        }
        request_payload = {
                "acceptance_criteria": signal_spec.acceptance_criteria,
                "rubric": rubric,
                "inputs": sample.inputs,
                "reference": sample.expected,
                "evaluation_context": context or {},
                "candidates": {
                    aliases[index]: outputs[index] for index in sorted(outputs)
                },
                "required_output": {
                    "candidate_judgments": required,
                    "preferred_candidate": "candidate_0",
                    "case_confidence": 0.0,
                },
            }
        payload: dict[str, Any] = {}
        responses: list[ModelResponse] = []
        missing: list[str] = list(aliases.values())
        for semantic_attempt in range(3):
            system = (
                "You are a strict comparative evaluation judge. Evaluate every anonymized "
                "candidate independently under the same rubric, then compare them. Do not improve "
                "answers. Return JSON only. You MUST include exactly one object under "
                f"candidate_judgments for every alias: {', '.join(aliases.values())}."
            )
            if semantic_attempt:
                system += (
                    f" Previous output was incomplete (missing: {', '.join(missing)}); "
                    "return the complete object from scratch."
                )
            payload = self.model.complete_json(
                system,
                json.dumps(request_payload, ensure_ascii=False),
            )
            response = getattr(self.model, "last_response", None)
            if response is not None:
                responses.append(response)
            values = payload.get("candidate_judgments", {})
            missing = [
                alias for alias in aliases.values()
                if not isinstance(values, dict) or not isinstance(values.get(alias), dict)
            ]
            if not missing:
                break
        if missing:
            raise ValueError(
                "Comparative Judge omitted candidate judgments after 3 attempts: "
                + ", ".join(missing)
            )
        response = responses[-1] if responses else None
        if response is not None and len(responses) > 1:
            response = ModelResponse(
                text=response.text,
                model=response.model,
                input_tokens=sum(item.input_tokens for item in responses),
                output_tokens=sum(item.output_tokens for item in responses),
                cost_usd=sum(item.cost_usd for item in responses),
                raw={**response.raw, "_promptos_semantic_attempts": len(responses)},
            )
            self.model.last_response = response
        values = payload["candidate_judgments"]
        judgments: dict[int, Judgment] = {}
        for index, alias in aliases.items():
            value = values.get(alias)
            if not isinstance(value, dict):
                raise ValueError(f"Comparative Judge omitted {alias}.")
            enriched = {
                **value,
                "_promptos_comparative": True,
                "_promptos_preferred_candidate": payload.get("preferred_candidate"),
                "_promptos_case_confidence": payload.get("case_confidence"),
            }
            judgments[index] = self._judgment(signal_spec, enriched, response)
        return judgments


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
        response = self.model._chat(
            "You improve prompts for a task. Return JSON only; do not evaluate any examples. "
            "Return exactly two distinct, complete, standalone candidate prompts. "
            "Do not echo the current prompt unchanged or merely normalize whitespace. "
            "Candidate 1 must materially strengthen ordered intent-routing and conflict precedence. "
            "Candidate 2 must materially strengthen ambiguity handling, confidence calibration, "
            "and output-schema self-checking.",
            json.dumps({
                "current_prompt": prompt,
                "task": asdict(task),
                "signal_spec": asdict(signal_spec),
                "feedback": feedback,
                "required_output": {
                    "candidates": [
                        "first complete standalone prompt",
                        "second complete standalone prompt",
                    ],
                },
            }, ensure_ascii=False),
        )
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        payload = json.loads(text)
        if isinstance(payload, dict):
            values = payload.get("candidates", [])
            if not values and isinstance(payload.get("current_prompt"), str):
                values = [payload["current_prompt"]]
        elif isinstance(payload, list):
            values = []
            for item in payload:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict):
                    values.extend(item.get("candidates", []))
                    if isinstance(item.get("current_prompt"), str):
                        values.append(item["current_prompt"])
        else:
            values = []
        candidates = [str(item) for item in values]
        normalized = {
            "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").splitlines()).strip()
            for value in candidates
        }
        current = "\n".join(
            line.rstrip() for line in prompt.replace("\r\n", "\n").splitlines()
        ).strip()
        fallbacks = [
            (
                f"{prompt.rstrip()}\n\n"
                "# 内部路由冲突检查（只执行，不输出检查过程）\n"
                "先识别用户最终要完成的动作，再按显式路由规则的优先级消解冲突；"
                "直接买卖/持仓操作意图优先于一般分析，代码、筛选、测算等任务执行意图"
                "优先于知识解释。选择 L3 后必须反查其 L2 父级，禁止跨父级组合。"
            ),
            (
                f"{prompt.rstrip()}\n\n"
                "# 内部歧义与输出自检（只执行，不输出检查过程）\n"
                "信息不足或存在多个同等合理意图时，采用保守分类并降低 confidence；"
                "不要用高置信度掩盖歧义。输出前检查 JSON 可解析、必需字段齐全、"
                "L2/L3 名称与 ID 一致、父子关系合法、dim_tag 规则正确、"
                "confidence 位于 0 到 1、reason 不超过 50 字，且不得输出额外文本。"
            ),
        ]
        for fallback in fallbacks:
            if len(normalized - {current}) >= 2:
                break
            key = "\n".join(
                line.rstrip() for line in fallback.replace("\r\n", "\n").splitlines()
            ).strip()
            if key != current and key not in normalized:
                candidates.append(fallback)
                normalized.add(key)
            if len(normalized - {current}) >= 2:
                break
        return candidates


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
