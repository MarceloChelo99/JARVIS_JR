"""Research: historical bars, macro release events, event studies, and the
point-in-time playbook the trading agent can consult.

bars          — CSV bar store + Tiingo fetcher
macro_events  — FRED/ALFRED vintages → first prints → surprise-labelled events
event_study   — conditional reaction stats (identity, no fitting)
playbook      — MacroPlaybook feed: lookup(series, direction, as_of)
"""

from jarvis_jr.research.bars import Bar, BarSeries, BarStore, TiingoBars
from jarvis_jr.research.event_study import EventStudy, ReactionStats, study
from jarvis_jr.research.macro_events import FredVintages, MacroEvent, macro_events
from jarvis_jr.research.playbook import MacroPlaybook, PlaybookEntry

__all__ = [
    "Bar",
    "BarSeries",
    "BarStore",
    "EventStudy",
    "FredVintages",
    "MacroEvent",
    "MacroPlaybook",
    "PlaybookEntry",
    "ReactionStats",
    "TiingoBars",
    "macro_events",
    "study",
]
