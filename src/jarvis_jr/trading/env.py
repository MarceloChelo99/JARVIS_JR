"""TradingEnv: a Gym-shaped (reset / step) view of a scenario.

Exists so that a learned policy and the LLM agent are evaluated on the SAME
scenarios with the SAME broker, guardrails and clock. No gym dependency —
the interface is the convention, not the package.

Action = a target exposure fraction in [0, 1], applied equal-weight across
the scenario's symbols by market orders that go through the usual
guardrails (so a policy cannot do anything the LLM agent couldn't). Reward =
log change in equity over the day that follows. Observation = the plain
numbers a small policy needs; nothing a feed wouldn't give the LLM.

# GROWTH: per-symbol weight vectors as actions when a policy needs them;
# today the learned component sizes risk, it does not pick names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from jarvis_jr.trading.domain import Broker, Limits
from jarvis_jr.trading.scenario import Scenario
from jarvis_jr.trading.tools import TradingTools


@dataclass
class Observation:
    day: date
    day_index: int
    equity: float
    cash: float
    exposure: float  # positions value / equity
    prices: dict[str, float]
    returns_1d: dict[str, float]  # today's close vs yesterday's, 0 on day 0
    event: dict = field(default_factory=dict)  # scenario.event block if generated
    is_event_day: bool = False


class TradingEnv:
    def __init__(self, scenario: Scenario, exposure_levels: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0)):
        self.scenario = scenario
        self.exposure_levels = exposure_levels
        self.broker: Broker | None = None
        self.tools: TradingTools | None = None
        self._day_index = 0
        self._prev: dict[str, float] = {}

    # ---- Gym-shaped API ---------------------------------------------------------

    def reset(self) -> Observation:
        self.broker = self.scenario.broker()
        self.tools = TradingTools(self.broker, self.scenario.limits)
        self._day_index = 0
        self._prev = {s: self.broker.quote(s).price for s in self.scenario.symbols}
        return self._observe()

    def step(self, action: float) -> tuple[Observation, float, bool, dict]:
        """Rebalance to `action` exposure, advance one day, return (obs, reward, done, info)."""
        assert self.broker is not None and self.tools is not None, "call reset() first"
        target = min(1.0, max(0.0, float(action)))
        before = self.broker.portfolio().equity
        rejections = self._rebalance(target)
        self._prev = {s: self.broker.quote(s).price for s in self.scenario.symbols}
        if self.broker.advance() is None:
            raise RuntimeError("episode is finished; call reset()")
        self._day_index += 1
        after = self.broker.portfolio().equity
        reward = math.log(after / before) if before > 0 and after > 0 else 0.0
        done = self._day_index >= len(self.scenario.days) - 1  # no day left to act on
        obs = self._observe()
        return obs, reward, done, {"rejections": rejections, "target_exposure": target}

    # ---- internals ---------------------------------------------------------------

    def _observe(self) -> Observation:
        p = self.broker.portfolio()
        prices = {s: self.broker.quote(s).price for s in self.scenario.symbols}
        rets = {s: (prices[s] / self._prev[s] - 1) if self._prev.get(s) else 0.0 for s in prices}
        event = getattr(self.scenario, "event", {}) or {}
        return Observation(
            day=p.day,
            day_index=self._day_index,
            equity=p.equity,
            cash=p.cash,
            exposure=(p.equity - p.cash) / p.equity if p.equity else 0.0,
            prices=prices,
            returns_1d=rets,
            event=event,
            is_event_day=bool(event) and event.get("day") == p.day.isoformat(),
        )

    def _rebalance(self, target: float) -> int:
        """Equal-weight the symbols to `target` of equity via guardrailed market orders.
        Sells first so the cash is there for buys. Returns the number of rejected orders."""
        assert self.broker is not None and self.tools is not None
        p = self.broker.portfolio()
        limits: Limits = self.scenario.limits
        per_symbol = min(target / len(self.scenario.symbols), limits.max_position_pct) if target else 0.0
        plans = []
        for s in self.scenario.symbols:
            price = self.broker.quote(s).price
            want_qty = math.floor(per_symbol * p.equity / price) if price else 0
            delta = want_qty - int(p.held(s))
            if delta:
                plans.append((s, delta))
        rejected = 0
        for s, delta in sorted(plans, key=lambda x: x[1]):  # sells (negative) first
            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)
            # respect the per-order notional cap by splitting
            price = self.broker.quote(s).price
            max_qty = max(1, math.floor(limits.max_order_notional / price)) if price else qty
            while qty > 0:
                chunk = min(qty, max_qty)
                out = self.tools.dispatch("place_order", {"symbol": s, "side": side, "qty": chunk})
                if out.startswith("ERROR"):
                    rejected += 1
                    break
                qty -= chunk
        return rejected

    def snap(self, exposure: float) -> float:
        """Nearest allowed exposure level."""
        return min(self.exposure_levels, key=lambda e: abs(e - exposure))


def run_policy(env: TradingEnv, policy, on_day=None) -> dict:
    """Roll a policy through a scenario. `policy(obs) -> exposure`. Returns a summary."""
    obs = env.reset()
    start = obs.equity
    done, total_reward, rejections, steps = False, 0.0, 0, 0
    exposures = []
    while not done:
        action = env.snap(policy(obs))
        exposures.append(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        rejections += info["rejections"]
        steps += 1
        if on_day:
            on_day(obs, action, reward)
    return {
        "days": steps,
        "start_equity": start,
        "end_equity": obs.equity,
        "return_pct": obs.equity / start - 1 if start else 0.0,
        "log_return": total_reward,
        "mean_exposure": sum(exposures) / len(exposures) if exposures else 0.0,
        "rejections": rejections,
    }
