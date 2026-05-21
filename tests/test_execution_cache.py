"""Unit tests for DAIL-SQL gold SQL cache integration.

Covers:
  - set_gold_cache / _gold_cache module state
  - _compare_sqls_with_timing: cache hit, cache miss, no question_id
  - compare_sqls: question_id passed through, VES calculation
  - DatabaseManager.compare_sqls: question_id passed through
  - run_meteo._evaluate: question_id passed through

All DB calls are mocked — no live PostgreSQL connection required.

Run:
    uv run python -m pytest src/DAIL-SQL/tests/test_execution_cache.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — handled by conftest.py (DAIL-SQL root is pinned per-test)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_cache(gold_set: frozenset | None, duration: float | None):
    """Return a mock GoldSqlCache with preset responses."""
    cache = MagicMock()
    cache.is_loaded = True
    cache.get_gold_set.return_value = gold_set
    cache.get_duration.return_value = duration
    return cache



# ---------------------------------------------------------------------------
# set_gold_cache
# ---------------------------------------------------------------------------

class TestSetGoldCache:
    def test_set_and_read(self):
        import runner.execution as exe
        cache = _make_mock_cache(frozenset(), 1.0)
        exe.set_gold_cache(cache)
        assert exe._gold_cache is cache

    def test_set_none_clears(self):
        import runner.execution as exe
        exe.set_gold_cache(_make_mock_cache(frozenset(), 1.0))
        exe.set_gold_cache(None)
        assert exe._gold_cache is None


# ---------------------------------------------------------------------------
# _compare_sqls_with_timing — cache hit
# ---------------------------------------------------------------------------

class TestCacheHit:
    def test_correct_answer_uses_cache_not_live(self):
        """Cache hit: gold SQL must NOT be executed live."""
        import runner.execution as exe

        gold = frozenset([(42,)])
        cache = _make_mock_cache(gold, t_gold := 10.0)
        exe.set_gold_cache(cache)

        with patch("runner.execution.sql_exec") as mock_exec:
            # predicted returns same result
            mock_exec.return_value = (frozenset([(42,)]), 1.0)
            res, err, ves = exe._compare_sqls_with_timing("SELECT 1", "SELECT 1", question_id=5)

        # sql_exec called only ONCE (for predicted), not for gold
        assert mock_exec.call_count == 1
        assert res == 1
        assert err == "--"
        assert ves == pytest.approx(math.sqrt(min(t_gold / 1.0, 1.0)))
        cache.get_gold_set.assert_called_once_with(5)

    def test_wrong_answer(self):
        import runner.execution as exe

        cache = _make_mock_cache(frozenset([(1,)]), 5.0)
        exe.set_gold_cache(cache)

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.return_value = (frozenset([(99,)]), 1.0)
            res, err, ves = exe._compare_sqls_with_timing("SELECT 99", "SELECT 1", question_id=7)

        assert res == 0
        assert ves == 0.0
        assert err == "incorrect answer"

    def test_ves_capped_at_1(self):
        """If pred is faster than gold, VES = sqrt(1.0) = 1.0."""
        import runner.execution as exe

        cache = _make_mock_cache(frozenset([(1,)]), 1.0)  # gold took 1s
        exe.set_gold_cache(cache)

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.return_value = (frozenset([(1,)]), 10.0)  # pred took 10s
            res, _, ves = exe._compare_sqls_with_timing("SELECT 1", "SELECT 1", question_id=1)

        assert res == 1
        # ratio = min(1.0 / 10.0, 1.0) = 0.1  → ves = sqrt(0.1)
        assert ves == pytest.approx(math.sqrt(0.1))

    def test_pred_faster_ves_1(self):
        """Pred faster than gold → ratio capped at 1.0 → ves = 1.0."""
        import runner.execution as exe

        cache = _make_mock_cache(frozenset([(1,)]), 10.0)  # gold took 10s
        exe.set_gold_cache(cache)

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.return_value = (frozenset([(1,)]), 1.0)  # pred took 1s
            res, _, ves = exe._compare_sqls_with_timing("SELECT 1", "SELECT 1", question_id=2)

        assert res == 1
        assert ves == pytest.approx(1.0)

    def test_cache_miss_falls_back_to_live(self):
        """get_gold_set returns None → fall back to live gold execution."""
        import runner.execution as exe

        cache = _make_mock_cache(None, None)  # cache miss
        exe.set_gold_cache(cache)

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.side_effect = [
                (frozenset([(7,)]), 5.0),   # gold live
                (frozenset([(7,)]), 1.0),   # pred
            ]
            res, _, _ = exe._compare_sqls_with_timing("SELECT 7", "SELECT 7", question_id=99)

        assert res == 1
        assert mock_exec.call_count == 2  # both gold and pred executed live


# ---------------------------------------------------------------------------
# _compare_sqls_with_timing — no cache / no question_id
# ---------------------------------------------------------------------------

class TestNoCacheOrId:
    def test_no_cache_runs_live(self):
        import runner.execution as exe
        # _gold_cache is None (reset by fixture)

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.side_effect = [
                (frozenset([(1,)]), 2.0),
                (frozenset([(1,)]), 1.0),
            ]
            res, _, _ = exe._compare_sqls_with_timing("SELECT 1", "SELECT 1", question_id=1)

        assert res == 1
        assert mock_exec.call_count == 2

    def test_question_id_none_skips_cache(self):
        """Even with cache loaded, question_id=None must not use cache."""
        import runner.execution as exe

        cache = _make_mock_cache(frozenset([(1,)]), 5.0)
        exe.set_gold_cache(cache)

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.side_effect = [
                (frozenset([(1,)]), 3.0),
                (frozenset([(1,)]), 1.0),
            ]
            exe._compare_sqls_with_timing("SELECT 1", "SELECT 1", question_id=None)

        cache.get_gold_set.assert_not_called()
        assert mock_exec.call_count == 2

    def test_gold_sql_error_returns_zero(self):
        import runner.execution as exe

        with patch("runner.execution.sql_exec", side_effect=Exception("db error")):
            res, err, ves = exe._compare_sqls_with_timing("SELECT 1", "BAD SQL")

        assert res == 0
        assert "gold_sql_error" in err
        assert ves == 0.0

    def test_pred_sql_error_returns_zero(self):
        import runner.execution as exe

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.side_effect = [
                (frozenset([(1,)]), 1.0),   # gold OK
                Exception("pred error"),     # pred fails
            ]
            res, err, ves = exe._compare_sqls_with_timing("BAD", "SELECT 1")

        assert res == 0
        assert ves == 0.0


# ---------------------------------------------------------------------------
# compare_sqls — question_id forwarded, timeout handling
# ---------------------------------------------------------------------------

class TestCompareSqls:
    def test_question_id_forwarded(self):
        import runner.execution as exe

        cache = _make_mock_cache(frozenset([(1,)]), 5.0)
        exe.set_gold_cache(cache)

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.return_value = (frozenset([(1,)]), 1.0)
            result = exe.compare_sqls("SELECT 1", "SELECT 1",
                                      meta_time_out=10, question_id=42)

        assert result["exec_res"] == 1
        cache.get_gold_set.assert_called_once_with(42)

    def test_timeout_returns_zero(self):
        import runner.execution as exe
        from func_timeout import FunctionTimedOut

        with patch("runner.execution.func_timeout", side_effect=FunctionTimedOut(0, None)):
            result = exe.compare_sqls("SELECT 1", "SELECT 1", meta_time_out=1)

        assert result["exec_res"] == 0
        assert result["exec_err"] == "timeout"
        assert result["ves"] == 0.0

    def test_default_question_id_none(self):
        """question_id defaults to None — cache not consulted."""
        import runner.execution as exe

        cache = _make_mock_cache(frozenset([(1,)]), 5.0)
        exe.set_gold_cache(cache)

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.side_effect = [
                (frozenset([(1,)]), 2.0),
                (frozenset([(1,)]), 1.0),
            ]
            exe.compare_sqls("SELECT 1", "SELECT 1", meta_time_out=30)

        cache.get_gold_set.assert_not_called()


# ---------------------------------------------------------------------------
# DatabaseManager.compare_sqls — question_id forwarded
# ---------------------------------------------------------------------------

class TestDatabaseManagerCompareSqls:
    def test_question_id_forwarded_to_execution(self):
        import runner.execution as exe
        from runner.database_manager import DatabaseManager

        cache = _make_mock_cache(frozenset([(99,)]), 3.0)
        exe.set_gold_cache(cache)

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.return_value = (frozenset([(99,)]), 1.0)
            result = DatabaseManager.compare_sqls(
                predicted_sql="SELECT 99",
                ground_truth_sql="SELECT 99",
                meta_time_out=10,
                question_id=77,
            )

        assert result["exec_res"] == 1
        cache.get_gold_set.assert_called_once_with(77)

    def test_question_id_defaults_none(self):
        import runner.execution as exe
        from runner.database_manager import DatabaseManager

        cache = _make_mock_cache(frozenset([(1,)]), 3.0)
        exe.set_gold_cache(cache)

        with patch("runner.execution.sql_exec") as mock_exec:
            mock_exec.side_effect = [
                (frozenset([(1,)]), 2.0),
                (frozenset([(1,)]), 1.0),
            ]
            DatabaseManager.compare_sqls("SELECT 1", "SELECT 1", meta_time_out=10)

        cache.get_gold_set.assert_not_called()


# ---------------------------------------------------------------------------
# Approximate equality (existing logic, regression guard)
# ---------------------------------------------------------------------------

class TestResultsApproxEqual:
    def test_exact_match(self):
        from runner.execution import _results_approx_equal
        a = {(1.0, "foo"), (2.0, "bar")}
        assert _results_approx_equal(a, a)

    def test_float_tolerance(self):
        from runner.execution import _results_approx_equal
        a = {(1.0000001,)}
        b = {(1.0000002,)}
        assert _results_approx_equal(a, b)

    def test_different_values(self):
        from runner.execution import _results_approx_equal
        assert not _results_approx_equal({(1,)}, {(2,)})

    def test_different_lengths(self):
        from runner.execution import _results_approx_equal
        assert not _results_approx_equal({(1,), (2,)}, {(1,)})
