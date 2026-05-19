"""
Shared LLM model adapter for OpenSearch-SQL and DAIL-SQL.

Uses the project's LLMFactory (config/llm_factory.py), supporting openai,
bedrock, scayle, and ollama providers.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# src/common/llm/model.py → parents[3] = project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import sys

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config.llm_factory import LLMFactory, _load_yaml  # type: ignore
except (ModuleNotFoundError, ImportError) as e:
    print(
        f"⚠️  config.llm_factory import failed ({e}), using fallback", file=sys.stderr
    )
    try:
        import yaml
    except ModuleNotFoundError:
        yaml = None

    def _load_yaml(path: Path) -> dict:
        if path.exists() and yaml is not None:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    return data
        env_model = (
            os.environ.get("MODEL")
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("LLM_MODEL")
            or "gpt-4o-mini"
        )
        return {
            "provider": os.environ.get("LLM_PROVIDER", "openai"),
            "model": env_model,
            "temperature": float(os.environ.get("LLM_TEMPERATURE", "0")),
        }

    class LLMFactory:
        @staticmethod
        def from_config_dict(config: dict[str, Any]):
            provider = str(config.get("provider", "openai")).lower()
            model = (
                config.get("model")
                or os.environ.get("MODEL")
                or os.environ.get("AWS_MODEL_ID")
                or os.environ.get("OPENAI_MODEL")
                or os.environ.get("LLM_MODEL")
                or "gpt-4o-mini"
            )
            temperature = float(config.get("temperature", 0))

            if provider == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model=model, temperature=temperature)
            elif provider == "bedrock":
                from langchain_aws import ChatBedrock
                region_name = config.get("region_name") or os.environ.get("AWS_REGION", "eu-west-1")
                return ChatBedrock(model_id=model, region_name=region_name, temperature=temperature)
            elif provider == "ollama":
                from langchain_ollama import ChatOllama
                base_url = config.get("base_url") or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                return ChatOllama(model=model, base_url=base_url, temperature=temperature)
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(model=model, temperature=temperature, api_key=os.environ.get("ANTHROPIC_API_KEY"))
            else:
                raise ValueError(f"Unsupported provider: {provider}. Use one of: openai, bedrock, ollama, anthropic")


from runner.logger import Logger
from llm.prompts import prompts_fewshot_parse

_LLM_CONFIG_PATH: Path = Path(
    os.environ.get("LLM_CONFIG_PATH", str(_PROJECT_ROOT / "config" / "models.yaml"))
)


def set_llm_config_path(path: str | Path) -> None:
    global _LLM_CONFIG_PATH
    _LLM_CONFIG_PATH = Path(path)


def _get_base_config() -> dict:
    return _load_yaml(_LLM_CONFIG_PATH)


def model_chose(step: str, model: str = "gpt-4 32K"):
    return LLMFactoryAdapter(step)


class req:
    def __init__(self, step: str, model: str = "") -> None:
        self.Cost = 0
        self.model = model
        self.step = step

    def log_record(self, prompt_text, output, duration_ms=None, model_info=None):
        try:
            logger = Logger()
            logger.log_conversation(prompt_text, "Human", self.step)
            logger.log_conversation(output, "AI", self.step)
            logger.log_llm_call(self.step, prompt_text, output, duration_ms=duration_ms, model_info=model_info)
        except (ValueError, AttributeError):
            pass

    def fewshot_parse(self, question, evidence, sql):
        s = prompts_fewshot_parse().parse_fewshot.format(question=question, sql=sql)
        ext = self.get_ans(s)
        ext = ext.replace("```", "").strip()
        ext = ext.split("#SQL:")[0]
        ans = self.convert_table(ext, sql)
        return ans

    def convert_table(self, s, sql):
        l = re.findall(r" ([^ ]*) +AS +([^ ]*)", sql)
        x, v = s.split("#values:")
        t, s = x.split("#SELECT:")
        for li in l:
            s = s.replace(f"{li[1]}.", f"{li[0]}.")
        return t + "#SELECT:" + s + "#values:" + v


class LLMFactoryAdapter(req):
    def __init__(self, step: str) -> None:
        super().__init__(step, model="llm_factory")
        self._base_config: dict = _get_base_config()
        self._llm_cache: dict[float, object] = {}

    @staticmethod
    def _parse_thinking(text: str) -> tuple[str, str]:
        match = re.search(r"<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", text

    def _get_llm(self, temperature: float | None = None):
        config = dict(self._base_config)
        if temperature is not None:
            config["temperature"] = temperature
        temp_key = config.get("temperature", 0.0)
        if temp_key not in self._llm_cache:
            self._llm_cache[temp_key] = LLMFactory.from_config_dict(config)
        return self._llm_cache[temp_key]

    def _build_messages(self, prompt_text: str):
        from langchain_core.messages import HumanMessage, SystemMessage
        system_base = "You are an SQL expert, skilled in handling various SQL-related issues."
        enable_thinking = self._base_config.get("enable_thinking", False)
        system_content = system_base if enable_thinking else f"{system_base} /no_think"
        return [SystemMessage(content=system_content), HumanMessage(content=prompt_text)]

    def get_ans(self, messages: str, temperature: float = 0.0, top_p: float | None = None,
                n: int = 1, single: bool = True, debug: bool = False, **kwargs):
        if debug:
            print(messages)
        llm = self._get_llm(temperature=temperature)
        chat_msgs = self._build_messages(messages)
        model_info = {"model": getattr(self, "_base_config", {}).get("model", "unknown"), "temperature": temperature, "n": n}
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                _t0 = time.time()
                if n == 1 or single:
                    result = llm.invoke(chat_msgs)
                    duration_ms = round((time.time() - _t0) * 1000, 1)
                    thinking, response_clean = self._parse_thinking(result.content)
                    if thinking:
                        model_info["thinking"] = thinking
                    if self.step != "prepare_train_queries":
                        self.log_record(messages, response_clean, duration_ms=duration_ms, model_info=model_info)
                    return response_clean
                else:
                    choices = self._generate_n(llm, chat_msgs, n)
                    duration_ms = round((time.time() - _t0) * 1000, 1)
                    for c in choices:
                        thinking, answer = self._parse_thinking(c["message"]["content"])
                        c["message"]["content"] = answer
                        if thinking:
                            c["thinking"] = thinking
                    if self.step != "prepare_train_queries":
                        self.log_record(messages, json.dumps([c["message"]["content"] for c in choices], ensure_ascii=False), duration_ms=duration_ms, model_info=model_info)
                    return choices
            except Exception as e:
                last_error = e
                wait = min(2 * attempt, 5)
                print(f"LLM error (attempt {attempt}/{max_retries}): {e}")
                time.sleep(wait)

        print(f"LLM failed after {max_retries} attempts. Last error: {last_error}")
        if n == 1 or single:
            return ""
        return [{"message": {"content": ""}} for _ in range(n)]

    def _generate_n(self, llm, chat_msgs, n: int):
        def _invoke(_):
            result = llm.invoke(chat_msgs)
            return {"message": {"content": result.content}}
        workers = min(n, 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_invoke, i) for i in range(n)]
            choices = [f.result() for f in futures]
        return choices
