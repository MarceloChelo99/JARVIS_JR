"""Agent evaluation harness: graded tasks, trajectories, reports.

Runs the assistant's tools (and the eval-side sandbox tools) against a local
model, then grades each run three ways: deterministic checks, tool-trajectory
match, and LLM-as-judge. See evals/tasks/ and scripts/run_evals.py.
"""

from jarvis_jr.evals.agent import Agent
from jarvis_jr.evals.domain import TaskReport, TaskSpec, Trajectory, Verdict
from jarvis_jr.evals.grading import ChecksGrader, JudgeGrader, TrajectoryGrader
from jarvis_jr.evals.model_lmstudio import LMStudioModel
from jarvis_jr.evals.run import run_suite, run_task
from jarvis_jr.evals.tasks import load_task, load_tasks
from jarvis_jr.evals.tools import available_tools, registry_tools

__all__ = [
    "Agent",
    "LMStudioModel",
    "TaskSpec",
    "TaskReport",
    "Trajectory",
    "Verdict",
    "ChecksGrader",
    "TrajectoryGrader",
    "JudgeGrader",
    "available_tools",
    "registry_tools",
    "load_task",
    "load_tasks",
    "run_suite",
    "run_task",
]
