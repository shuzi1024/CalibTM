"""CPU-only policy and interprocess scheduling checks; no torch or GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from . import watch


def row(variant, abilene, geant, latency=1.0):
    return {"variant": variant, "relative_gains": {"abilene": abilene, "geant": geant},
            "mean_relative_gain": (abilene + geant) / 2, "mean_normalized_latency": latency}


class PolicyChecks(unittest.TestCase):
    def test_both_wans_preferred_to_larger_mixed_mean(self):
        winner, reason, _ = watch.choose_candidate([
            row("routing_sync", .001, .001, 3), row("direct_sync", -.02, .20, .5)])
        self.assertEqual((winner, reason), ("routing_sync", "both_wans_nonregressing"))

    def test_mixed_mean_labeled_uncertain(self):
        self.assertEqual(watch.choose_candidate([row("direct_sync", -.01, .03)])[:2],
                         ("direct_sync", "uncertain_one_wan_regression"))

    def test_no_positive_mean_no_repeat(self):
        self.assertIsNone(watch.choose_candidate([row("direct_sync", -.02, .02),
                                                  row("feature_sync", -.01, -.01)])[0])

    def test_near_tie_prefers_task_contribution_before_latency(self):
        self.assertEqual(watch.choose_candidate([row("routing_sync", .016, .016, 2),
                                                row("direct_sync", .02, .02, 1)])[0], "routing_sync")
        self.assertEqual(watch.choose_candidate([row("direct_sync", .02, .02),
                                                row("routing_sync", .02, .02)])[0], "routing_sync")
        self.assertEqual(watch.choose_candidate([row("feature_sync", .02, .02, .5),
                                                row("direct_sync", .016, .016, 2)])[0], "direct_sync")

    def test_half_percentage_point_boundary_is_inclusive(self):
        winner, _, near_tied = watch.choose_candidate([
            row("routing_sync", .015, .015, 3), row("direct_sync", .02, .02, .1)])
        self.assertEqual(winner, "routing_sync")
        self.assertEqual(set(near_tied), {"routing_sync", "direct_sync"})

    def test_outside_tie_prefers_score(self):
        self.assertEqual(watch.choose_candidate([row("routing_sync", .0149, .0149, .1),
                                                row("direct_sync", .02, .02, 3)])[0], "direct_sync")

    def test_novelty_preference_never_overrides_positive_gain_and_two_wan_gate(self):
        self.assertEqual(watch.choose_candidate([row("routing_sync", 0, 0, .1),
                                                row("feature_sync", .001, .001, 3)])[0], "feature_sync")
        self.assertEqual(watch.choose_candidate([row("routing_sync", -.001, .022, .1),
                                                row("direct_sync", .01, .01, 3)])[0], "direct_sync")


class QueueChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="calibtm-watch-check-")
        self.out = Path(self.temporary.name)
        references = {}
        for dataset in watch.DATASETS:
            directory = self.out / "references" / dataset
            result = directory / "result.json"
            watch.atomic_json(result, {"fixture": True, "score": 1.0})
            watch.atomic_json(directory / "latency.json", {"median_ms": 10.0})
            references[dataset] = {"sync_delta": {"path": str(result), "sha256": watch.sha(result),
                                                  "selection_score": 1.0}}
        old_repeat = self.out / "references/geant/repeat_result.json"
        watch.atomic_json(old_repeat, {"fixture": True, "seed": 41002})
        self.registry = {"deadline_unix": watch.DEADLINE_UNIX, "screen_variants": list(watch.VARIANTS),
                         "maximum_formal_new_jobs": 9, "references": references,
                         "existing_geant_sync_repeat": {"path": str(old_repeat), "sha256": watch.sha(old_repeat),
                                                        "seed": 41002, "selection_score": .9},
                         "initializations": {d: {v: {str(s): {"state_sha256": f"{d}-{v}-{s}"}
                             for s in (41001, 41002)} for v in (*watch.VARIANTS, "sync_delta")}
                             for d in watch.DATASETS}}
        watch.atomic_json(self.out / "registry.json", self.registry)
        self.digest = watch.sha(self.out / "registry.json")
        (self.out / "registry.sha256").write_text(self.digest + "\n")
        self.leases = []
        self.owner = {"node_id": "fixture", "hostname": "fixture-host", "gpu": 0}

    def tearDown(self):
        for lease in self.leases:
            lease.close()
        self.temporary.cleanup()

    def claim(self, now=None):
        claimed, all_done = watch.claim(self.out, self.registry, self.digest, self.owner,
                                       watch.ALLOCATION_CUTOFF - 3600 if now is None else now)
        if claimed:
            self.leases.append(claimed[1])
        return claimed, all_done

    def complete(self, job, score=.9, latency=10.0, code=0):
        directory = watch.job_dir(self.out, job)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "best.pt").write_bytes(b"not a tensor: scheduling fixture only")
        watch.atomic_json(directory / "latency.json", {"median_ms": latency})
        watch.atomic_json(directory / "result.json", {
            "registry_sha256": self.digest, "dataset": job["dataset"], "variant": job["variant"],
            "seed": job["seed"], "initial_state_sha256": self.registry["initializations"][job["dataset"]][job["variant"]][str(job["seed"])]["state_sha256"],
            "role": "development_only", "best_checkpoint_sha256": watch.sha(directory / "best.pt"),
            "primary_eval": {"role": "development_only", "mask_seed": 71001, "selection_score": score}})
        watch.atomic_json(directory / "worker_exit.json", {"registry_sha256": self.digest,
            "returncode": code, "finished": watch.utc(), "stage": "train"})

    def test_six_unique_claims_in_fixed_priority(self):
        self.assertEqual(watch.verify_registry(self.out)[1], self.digest)
        actual = [self.claim()[0][0]["id"] for _ in range(6)]
        self.assertEqual(actual, [j["id"] for j in watch.screen_jobs()])
        self.assertIsNone(self.claim()[0])
        self.assertFalse((self.out / "decision.json").exists())

    def test_child_inherits_lease_and_blocks_duplicate_claim(self):
        job, lease = self.claim()[0]
        gpu_path = self.out / "scheduler/workers/gpu_fixture_0.lock"
        gpu_lease = watch.try_lock(gpu_path)
        self.leases.append(gpu_lease)
        process = subprocess.Popen([sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.readline()"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                   pass_fds=(lease.fileno(), gpu_lease.fileno()))
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            lease.close()  # Simulate worker death while the training child lives.
            gpu_lease.close()
            self.assertIsNone(watch.try_lock(watch.lease_path(self.out, job)))
            self.assertIsNone(watch.try_lock(gpu_path))
            other, _ = self.claim()
            self.assertNotEqual(other[0]["id"], job["id"])
        finally:
            process.communicate("finish\n", timeout=5)
        reclaimed, _ = self.claim()
        self.assertEqual(reclaimed[0]["id"], job["id"])
        self.assertEqual(len(reclaimed[0]["attempts"]), 2)

    def test_unselected_repeat_in_queue_is_rejected(self):
        self.claim()
        queue = watch.read_json(self.out / "scheduler/queue.json")
        queue["jobs"].append(watch.identity("geant", "sync_delta", 41002, "repeat"))
        watch.atomic_json(self.out / "decision.json", {"registry_sha256": self.digest, "winner": "direct_sync"})
        watch.save_queue(self.out, queue)
        with self.assertRaisesRegex(RuntimeError, "unselected repeat"):
            watch.load_queue(self.out, self.digest)

    def test_resume_flag_only_for_same_existing_job(self):
        job = watch.screen_jobs()[0]
        self.assertNotIn("--resume", watch.command(self.out, job, "train"))
        directory = watch.job_dir(self.out, job)
        watch.atomic_json(directory / "run_info.json", {"fixture": True})
        self.assertIn("--resume", watch.command(self.out, job, "train"))
        self.assertNotIn("--resume", watch.command(self.out, job, "smoke"))
        self.assertNotIn("--resume", watch.command(self.out, watch.screen_jobs()[1], "train"))

    def test_existing_result_not_retrained(self):
        job = watch.screen_jobs()[0]
        self.complete(job)
        selected, _ = self.claim()
        self.assertNotEqual(selected[0]["id"], job["id"])
        queue = watch.read_json(self.out / "scheduler/queue.json")
        self.assertEqual(queue["jobs"][0]["state"], "complete")

    def test_failed_exit_never_automatically_restarted(self):
        job = watch.screen_jobs()[0]
        watch.atomic_json(watch.job_dir(self.out, job) / "worker_exit.json", {
            "registry_sha256": self.digest, "returncode": 1, "finished": watch.utc()})
        selected, _ = self.claim()
        self.assertNotEqual(selected[0]["id"], job["id"])
        self.assertEqual(watch.read_json(self.out / "scheduler/queue.json")["jobs"][0]["state"], "failed")

    def test_decision_waits_for_six_terminal_then_only_three_repeats(self):
        for job in watch.screen_jobs()[:-1]:
            self.complete(job, score=.9 if job["variant"] == "direct_sync" else .95)
        selected, _ = self.claim()
        self.assertEqual(selected[0]["id"], watch.screen_jobs()[-1]["id"])
        self.assertFalse((self.out / "decision.json").exists())
        self.complete(selected[0], score=.95)
        selected[1].close()
        repeated, _ = self.claim()
        decision = watch.read_json(self.out / "decision.json")
        self.assertEqual(decision["winner"], "direct_sync")
        self.assertEqual(decision["registry_sha256"], self.digest)
        self.assertEqual(decision["selection_preference"], "balanced_slight_novelty")
        self.assertEqual(decision["tie_gain_threshold"], .005)
        self.assertEqual(decision["novelty_priority"], ["routing_sync", "direct_sync", "feature_sync"])
        self.assertEqual(repeated[0]["seed"], 41002)
        queue = watch.read_json(self.out / "scheduler/queue.json")
        self.assertEqual(len(queue["jobs"]), 9)
        self.assertEqual({j["id"] for j in queue["jobs"] if j["phase"] == "repeat"}, {
            "geant/direct_sync/seed41002", "abilene/direct_sync/seed41002", "abilene/sync_delta/seed41002"})
        self.assertNotIn("geant/sync_delta/seed41002", {j["id"] for j in queue["jobs"]})

    def test_one_wan_failure_excluded_others_still_compared(self):
        for job in watch.screen_jobs():
            failed = job["variant"] == "routing_sync" and job["dataset"] == "geant"
            score = .5 if job["variant"] == "routing_sync" else (.9 if job["variant"] == "direct_sync" else .95)
            self.complete(job, score=score, code=1 if failed else 0)
        self.claim()
        decision = watch.read_json(self.out / "decision.json")
        self.assertEqual(decision["winner"], "direct_sync")
        self.assertEqual(decision["excluded_candidates"]["routing_sync"]["geant"], "failed")

    def test_deadline_stops_allocations_and_marks_unstarted_incomplete(self):
        selected, all_done = self.claim(now=watch.ALLOCATION_CUTOFF)
        self.assertIsNone(selected)
        self.assertTrue(all_done)
        queue = watch.read_json(self.out / "scheduler/queue.json")
        self.assertEqual(len(queue["jobs"]), 6)
        self.assertTrue(all(j["state"] == "incomplete" for j in queue["jobs"]))
        self.assertIsNone(watch.read_json(self.out / "decision.json")["winner"])

    def test_signal_stops_claims_and_process_launches(self):
        previous = watch.STOP_REQUESTED
        watch.STOP_REQUESTED = True
        try:
            self.assertIsNone(self.claim()[0])
            with self.assertRaises(watch.AllocationStopped):
                watch.launch(self.out, watch.screen_jobs()[0], None, 0, "smoke")
        finally:
            watch.STOP_REQUESTED = previous

    def test_nonpositive_scores_do_not_force_repeat(self):
        for job in watch.screen_jobs():
            self.complete(job, score=1.02)
        selected, all_done = self.claim()
        self.assertIsNone(selected)
        self.assertTrue(all_done)
        self.assertEqual(len(watch.read_json(self.out / "scheduler/queue.json")["jobs"]), 6)
        self.assertIsNone(watch.read_json(self.out / "decision.json")["winner"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (PolicyChecks, QueueChecks)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "torch_imported": "torch" in sys.modules,
              "scope": "CPU policy, identity, resume and inherited interprocess flock checks; no GPU/training jobs."}
    if args.output:
        watch.atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
