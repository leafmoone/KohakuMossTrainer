import pytest

from kohakumoss.training.callbacks.runner import CallbackRunner, EpochCallback
from kohakumoss.training.checks.objective import schedule_problems
from kohakumoss.training.checks.parallel import _pipeline_problems
from kohakumoss.training.epochs import epoch_steps
from kohakumoss.training.state import TrainState

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


class Planner:
    balance = 0

    def __init__(self, lengths):
        self.lengths = lengths

    def epoch_length(self, epoch):
        return self.lengths[epoch]


def test_epoch_steps_use_each_actual_epoch_length():
    assert epoch_steps(Planner([8, 12, 4]), 3, 2, 2) == (2, 5, 6)


def test_epoch_steps_reject_unrepresentable_boundaries():
    with pytest.raises(ValueError, match="epoch 1"):
        epoch_steps(Planner([5, 7]), 2, 2, 2)
    with pytest.raises(ValueError, match="empty"):
        epoch_steps(Planner([0]), 1, 1, 1)
    planner = Planner([16])
    planner.balance = 4
    with pytest.raises(ValueError, match="balancing"):
        epoch_steps(planner, 1, 2, 2)


def test_schedule_checks_use_resolved_epoch_steps():
    k = {"EPOCHS": 20, "STEPS": 50, "SCHEDULE": {"mode": "cosine", "warmup": 100}}
    assert not schedule_problems(k)
    assert not schedule_problems({**k, "STEPS": 320, "_EPOCH_BOUNDARIES": (320,)})
    assert schedule_problems({**k, "STEPS": 50, "_EPOCH_BOUNDARIES": (50,)})


def test_trace_checks_use_resolved_epoch_steps():
    k = {
        "EPOCHS": 20,
        "STEPS": 50,
        "PIPELINE_SPLIT": "even",
        "TRACE_STEPS": (10, 200),
        "PROFILE_STEPS": (),
    }
    assert not _pipeline_problems(k)
    assert not _pipeline_problems({**k, "STEPS": 320, "_EPOCH_BOUNDARIES": (320,)})
    assert _pipeline_problems({**k, "STEPS": 100, "_EPOCH_BOUNDARIES": (100,)})


def test_epoch_sampling_and_resume():
    class Sample:
        every = 200
        at_start = True

        def __init__(self):
            self.calls = []

        def on_step_end(self, state, out):
            self.calls.append(state.step)

    for start, expected in [(0, [0, 3, 7, 12]), (7, [12])]:
        sample = Sample()
        runner = CallbackRunner([EpochCallback(sample, (3, 7, 12))], start_step=start)
        state = TrainState(step=start)
        runner.start(state)
        for step in range(start + 1, 13):
            state.step = step
            runner.run(state, None)
        assert sample.calls == expected


@pytest.mark.parametrize(
    ("start", "at_start", "expected"),
    [(0, True, [0, 7, 18]), (0, False, [7, 18]), (7, True, [18]), (12, True, [18])],
)
def test_epoch_sampling_interval_uses_epoch_number(start, at_start, expected):
    class Sample:
        every = 200

        def __init__(self):
            self.at_start = at_start
            self.calls = []

        def on_step_end(self, state, out):
            self.calls.append(state.step)

    sample = Sample()
    callback = EpochCallback(sample, (3, 7, 12, 18, 25), every_epochs=2)
    runner = CallbackRunner([callback], start_step=start, last=25)
    state = TrainState(step=start)
    runner.start(state)
    for step in range(start + 1, 26):
        state.step = step
        runner.run(state, None)
    assert sample.calls == expected


def test_epoch_sampling_and_rank_barrier_fire_together():
    calls = []

    class Sample:
        every = 200
        at_start = True

        def __init__(self, label):
            self.label = label

        def on_step_end(self, state, out):
            calls.append((self.label, state.step))

    boundaries = (3, 7, 12, 18, 25)
    runner = CallbackRunner(
        [
            EpochCallback(Sample(label), boundaries, every_epochs=2)
            for label in ("sample", "barrier")
        ]
    )
    state = TrainState(step=0)
    runner.start(state)
    for step in range(1, 26):
        state.step = step
        runner.run(state, None)
    assert calls == [
        (label, step) for step in (0, 7, 18) for label in ("sample", "barrier")
    ]


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2", None])
def test_epoch_sampling_interval_rejects_invalid_values(value, monkeypatch):
    from kohakumoss.training.checks import outputs

    monkeypatch.setattr(outputs, "export_problems", lambda k: [])
    monkeypatch.setattr(outputs, "weights_problems", lambda k: [])
    k = {
        "SAMPLE_EVERY": 0,
        "SAMPLE_PROMPTS": [],
        "SAMPLE_EVERY_EPOCHS": value,
        "VALIDATE_EVERY": 0,
        "SAVE_EVERY": 0,
        "KEEP_LAST": 0,
        "RESUME": "",
    }
    assert any("SAMPLE_EVERY_EPOCHS" in error for error in outputs.io_problems(k))
