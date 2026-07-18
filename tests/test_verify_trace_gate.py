from library_learning import verify
from library_learning.config import resolve_target

IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"


def test_trace_crash_forces_mismatch(monkeypatch):
    """A crashed trace check must never be masked into a MATCH verdict."""
    target = resolve_target(IND)

    def boom(*args, **kwargs):
        raise RuntimeError("instrumentation exploded")

    monkeypatch.setattr(verify.instrument, "exec_traced", boom)
    # baseline=True: lib == orig, so every other check trivially passes;
    # only the trace crash can determine the verdict here.
    rec = verify.verify_participant(target, 14, probes=2, synthetic=1, baseline=True)
    assert rec["trace_available"] is False
    assert "trace_error" in rec
    assert rec["status"] == "MISMATCH"
