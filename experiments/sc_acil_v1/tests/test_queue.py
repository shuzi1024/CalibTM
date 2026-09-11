from __future__ import annotations


def test_queue_slots_repeat_each_gpu_without_changing_job_grid() -> None:
    from experiments.sc_acil_v1.gpu_queue import gpu_slots

    assert gpu_slots(("0", "1", "2", "3"), 1) == ("0", "1", "2", "3")
    assert gpu_slots(("0", "1"), 3) == ("0", "0", "0", "1", "1", "1")

