"""ADR 0008 Decision 5: the bare entry point is 1 x 1 with 2 torch threads; more than one
engine instance is refused, before any process starts, unless --allow-multi-instance
equals the instance count. No engine is ever started here."""
import os

import psutil
import pytest

import a0_helpers  # noqa: F401
from configs.train_config import PPOConfig, RunConfig, SafetyConfig
from train import run_doom
from train.doom_factory import DoomEnvFactory
from train.layout_guard import LayoutRefused, check_layout, count_engine_instances
from train.subproc_vec_env import SubprocVecEnv
from train.vec_env import SyncVecEnv


class Sentinel(Exception):
    pass


@pytest.fixture
def no_engines(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub everything that would start a process; record what main() asked for."""
    seen: dict = {"venv_calls": [], "engine_checks": 0}

    def fake_engine_processes(names):
        seen["engine_checks"] += 1
        return []

    def fake_venv(env_fns_per_worker, safety, allow_multi_instance=None, torch_threads=0):
        seen["venv_calls"].append({"layout": [len(f) for f in env_fns_per_worker],
                                   "allow": allow_multi_instance, "threads": torch_threads})
        raise Sentinel  # stop before any process would start

    monkeypatch.setattr(run_doom.PC, "engine_processes", fake_engine_processes)
    monkeypatch.setattr(run_doom.PC, "set_below_normal", lambda pid=None: None)
    monkeypatch.setattr(run_doom, "SubprocVecEnv", fake_venv)
    return seen


def test_defaults_are_one_instance_two_threads() -> None:
    r = RunConfig()
    assert (r.num_workers, r.envs_per_worker, r.torch_threads, r.num_envs) == (1, 1, 2, 1)


def test_bare_command_resolves_to_1x1(no_engines: dict) -> None:
    run = run_doom.build_run_config(run_doom.parse_args([]))
    assert (run.num_workers, run.envs_per_worker, run.torch_threads) == (1, 1, 2)
    assert run.ppo.num_minibatches == 1 and run.num_envs % run.ppo.num_minibatches == 0
    with pytest.raises(Sentinel):
        run_doom.main([])
    assert no_engines["venv_calls"] == [{"layout": [1], "allow": None, "threads": 2}]


def test_minibatch_default_divides_env_count() -> None:
    assert run_doom.default_minibatches(1, 4) == 1
    assert run_doom.default_minibatches(8, 4) == 4
    assert run_doom.default_minibatches(6, 4) == 3
    assert run_doom.default_minibatches(5, 4) == 1
    assert PPOConfig().num_minibatches == 4


@pytest.mark.parametrize("argv", [["--workers", "2"], ["--envs-per-worker", "2"],
                                  ["--workers", "2", "--envs-per-worker", "4", "--allow-multi-instance", "4"],
                                  ["--workers", "2", "--envs-per-worker", "4", "--allow-multi-instance", "1"]])
def test_multi_instance_without_matching_flag_refused_before_any_process(
        no_engines: dict, argv: list[str], capsys: pytest.CaptureFixture) -> None:
    children_before = {p.pid for p in psutil.Process().children(recursive=True)}
    assert run_doom.main(argv) == 4
    assert no_engines["venv_calls"] == [] and no_engines["engine_checks"] == 0
    assert {p.pid for p in psutil.Process().children(recursive=True)} <= children_before
    err = capsys.readouterr().err
    assert "planned layout" in err and "torch thread" in err and "estimated memory" in err


def test_multi_instance_with_exact_flag_passes_the_guard(no_engines: dict) -> None:
    with pytest.raises(Sentinel):
        run_doom.main(["--workers", "2", "--envs-per-worker", "4", "--allow-multi-instance", "8"])
    assert no_engines["venv_calls"] == [{"layout": [4, 4], "allow": 8, "threads": 2}]


def test_entry_point_refuses_when_engines_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    started = {"venv": False}
    monkeypatch.setattr(run_doom.PC, "engine_processes", lambda names: [psutil.Process()])
    monkeypatch.setattr(run_doom, "SubprocVecEnv", lambda *a, **k: started.__setitem__("venv", True))
    assert run_doom.main([]) == 2
    assert not started["venv"]


def test_guard_lives_in_the_vector_envs_and_counts_engines_not_toys() -> None:
    factory = DoomEnvFactory(config=None)   # never called: the guard refuses first
    assert count_engine_instances([factory, factory, lambda: None]) == 2
    with pytest.raises(LayoutRefused):
        SyncVecEnv([factory, factory])                       # two engines in one process
    with pytest.raises(LayoutRefused):
        SyncVecEnv([factory, factory], allow_multi_instance=3)
    children_before = {p.pid for p in psutil.Process().children(recursive=True)}
    with pytest.raises(LayoutRefused):
        SubprocVecEnv([[factory], [factory]], SafetyConfig())
    with pytest.raises(LayoutRefused):
        SubprocVecEnv([[factory, factory]], SafetyConfig(), allow_multi_instance=1)
    assert {p.pid for p in psutil.Process().children(recursive=True)} <= children_before
    check_layout(1, None, 1, 2, SafetyConfig())
    check_layout(0, None, 4, 2, SafetyConfig())             # toy envs: no engines
    check_layout(8, 8, 2, 2, SafetyConfig())


def test_toy_envs_do_not_count() -> None:
    from functools import partial

    from train.toy_memory import ToyConfig, ToyMemoryEnv
    v = SyncVecEnv([partial(ToyMemoryEnv, ToyConfig(answer_step=3)) for _ in range(3)])
    v.close()
    assert os.environ.get("PYTEST_CURRENT_TEST")
