import math
from datetime import date
from pathlib import Path

import pytest

from jarvis_jr.trading.domain import Limits
from jarvis_jr.trading.env import TradingEnv, run_policy
from jarvis_jr.trading.scenario import Expectations, Scenario, load_scenario

DAYS = 6


def _scenario(prices, cash=100_000.0, max_pos=0.6, notional=1e9):
    return Scenario(
        id="t", start_day=date(2026, 1, 5), cash=cash, prices=prices,
        limits=Limits(max_position_pct=max_pos, max_order_notional=notional, allowed_symbols=frozenset(prices)),
        expect=Expectations(),
    )


def test_reset_step_reward_and_done():
    env = TradingEnv(_scenario({"A": [100, 110, 121, 121, 121, 121]}))
    obs = env.reset()
    assert obs.day_index == 0 and obs.exposure == 0.0 and obs.returns_1d == {"A": 0.0}
    obs, reward, done, info = env.step(1.0)  # all in (capped at 60% by max_position_pct)
    assert not done and info["rejections"] == 0
    assert obs.exposure == pytest.approx(0.6 * 1.1 / (0.4 + 0.6 * 1.1), rel=1e-3)
    assert reward == pytest.approx(math.log(0.4 + 0.6 * 1.1))
    assert obs.returns_1d["A"] == pytest.approx(0.10)
    rewards = [reward]
    while not done:
        obs, reward, done, _ = env.step(0.0)
        rewards.append(reward)
    assert len(rewards) == DAYS - 1  # one step per advance; the last day cannot be acted on
    assert obs.exposure == 0.0  # sold out
    with pytest.raises(RuntimeError):
        env.step(0.0)


def test_rebalance_respects_guardrails_and_splits_orders():
    env = TradingEnv(_scenario({"A": [100.0] * DAYS, "B": [50.0] * DAYS}, max_pos=0.25, notional=5_000))
    env.reset()
    obs, _, _, info = env.step(1.0)
    # 1.0 equal-weight -> 0.5 each, capped to 0.25 each; orders split into <=5000 chunks
    assert info["rejections"] == 0
    assert obs.exposure == pytest.approx(0.5, abs=0.01)
    assert len(env.tools.placed) >= 10  # 25k per symbol / 5k chunks


def test_run_policy_baselines():
    env = TradingEnv(_scenario({"A": [100, 90, 80, 85, 90, 95]}))
    flat = run_policy(env, lambda obs: 0.0)
    assert flat["return_pct"] == 0.0 and flat["mean_exposure"] == 0.0
    long = run_policy(env, lambda obs: 1.0)
    assert long["return_pct"] < 0 and long["mean_exposure"] == 1.0
    reactive = run_policy(env, lambda obs: 0.0 if obs.returns_1d["A"] < 0 else 1.0)
    assert flat["return_pct"] <= reactive["return_pct"] or reactive["return_pct"] > long["return_pct"]


def test_generated_scenario_event_block_reaches_observation():
    families = Path(__file__).resolve().parents[2] / "evals" / "scenarios" / "generated" / "cpi_release"
    paths = sorted(families.rglob("scenario.yaml")) if families.is_dir() else []
    if not paths:
        pytest.skip("no generated scenarios on this machine")
    env = TradingEnv(load_scenario(paths[0]))
    obs = env.reset()
    assert obs.event["type"] == "cpi_release" and not obs.is_event_day  # event is at index before_days
    for _ in range(obs.event["event_index_in_window"]):
        obs, *_ = env.step(0.0)
    assert obs.is_event_day
