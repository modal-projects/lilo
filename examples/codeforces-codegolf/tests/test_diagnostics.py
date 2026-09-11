from codegolf.diagnostics import current_records


def test_diagnostics_selects_consumed_batch_after_rollback():
    old = [{"sampling_ticket": 351, "rows": [0] * 8} for _ in range(4)]
    current = [{"sampling_ticket": 352, "rows": [1] * 8} for _ in range(4)]
    mixed = old + current
    selected = current_records(mixed, {"pipeline": {"sampling_ticket": 352}})
    assert selected == current
    assert sum(len(r["rows"]) for r in selected) == 32
    assert current_records(mixed, {}) == mixed
