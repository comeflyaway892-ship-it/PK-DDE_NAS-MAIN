"""CPU checks with a synthetic API; these are not benchmark results."""

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_component_ablation as runner
from prior_exp.predictor_prior import BayesianEstimator


class FakeInfo:
    def __init__(self, api, index):
        self.api, self.index = api, index

    def get_metrics(self, dataset, split, is_random):
        assert dataset == runner.DATASET and is_random is False
        self.api.reads.append((self.index, split))
        # Validation and test rank architectures in opposite order.
        score = 20.0 + self.index / 1000
        return {"accuracy": score if split == "x-valid" else 100.0 - score}


class FakeAPI:
    def __init__(self):
        self.arch2infos_dict, self.reads, self.proposals = {}, [], []

    def query_index_by_arch(self, arch):
        names = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
        tokens = [names.index(piece.split("~")[0]) for piece in arch.replace("+", "").split("|") if piece]
        index = sum(token * 5**edge for edge, token in enumerate(tokens))
        self.proposals.append(tokens)
        self.arch2infos_dict[index] = {"200": FakeInfo(self, index)}
        return index


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_six_have_exact_budget_and_use_only_test_accuracy(self):
        args = runner.parse_args(["--runs", "1", "--log-every", "0"])
        initials = []
        for variant in runner.VARIANTS:
            api = FakeAPI()
            result = runner.run_variant(args, api, variant, seed=7)
            self.assertEqual(result["architecture_evaluations"], 320)
            self.assertEqual(len(result["history"]), 15)
            self.assertEqual(result["validity_pct"], 100)
            self.assertTrue(all(split == "x-test" for _, split in api.reads))
            self.assertEqual(result["unique_architectures"], len(api.reads))
            self.assertEqual(result["final_test_queries"], 0)
            self.assertEqual(result["search_split"], "x-test")
            self.assertEqual(result["test_accuracy"], result["best_test_accuracy"])
            self.assertAlmostEqual(result["test_accuracy"], 80.0 - result["best_index"] / 1000)
            self.assertEqual(result["best_index"], min(index for index, _ in api.reads))
            self.assertNotIn("best_valid_accuracy", result)
            self.assertEqual(result["history"][-1]["best_test_accuracy"], result["test_accuracy"])
            initials.append(api.proposals[:20])
        self.assertTrue(all(x == initials[0] for x in initials))
        self.assertEqual(runner.settings(args)["search_split"], "x-test")
        self.assertEqual(runner.settings(args)["final_test_queries_per_run"], 0)

    def test_duplicates_are_charged_and_budget_is_enforced(self):
        api = FakeAPI()
        evaluator = runner.BudgetEvaluator(api, 3)
        evaluator.evaluate(torch.ones((3, 6), dtype=torch.long))
        self.assertEqual(evaluator.count, 3)
        self.assertEqual(len(api.reads), 1)
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            evaluator.evaluate(torch.ones((1, 6), dtype=torch.long))

    def test_missing_metric_fails_instead_of_becoming_invalid(self):
        api = FakeAPI()
        with patch.object(FakeInfo, "get_metrics", side_effect=KeyError("missing data")):
            with self.assertRaises(KeyError):
                runner.BudgetEvaluator(api, 1).evaluate(torch.ones((1, 6), dtype=torch.long))

    def test_validity_and_connectivity_are_separate(self):
        api = FakeAPI()
        x = torch.tensor([[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1], [-1, 0, 0, 0, 0, 0]])
        _, _, validity, connectivity = runner.BudgetEvaluator(api, 3).evaluate(x)
        self.assertAlmostEqual(validity, 200 / 3)
        self.assertAlmostEqual(connectivity, 100 / 3)

    def test_full_estimator_matches_existing_prior_estimator(self):
        args = runner.parse_args([])
        torch.manual_seed(4)
        x = torch.randint(0, 5, (20, 6))
        fitness = torch.rand(20)
        scheduler = runner.make_scheduler(args, [x[:6]])
        estimator = BayesianEstimator(x, fitness, scheduler.Q, scheduler.Qbar, 15,
                                      eps=runner.cfg.predictor_estimator_eps)
        torch.manual_seed(99)
        expected = estimator(sigma=args.sigma, temperature=args.temperature)
        torch.manual_seed(99)
        actual = runner.estimate_x0(x, fitness, scheduler, 15, args)
        self.assertTrue(torch.equal(expected, actual))

    def test_disabled_guidance_ignores_fitness_in_estimator(self):
        args = runner.parse_args([])
        x = torch.randint(0, 5, (20, 6))
        scheduler = runner.make_scheduler(args)
        torch.manual_seed(5)
        a = runner.estimate_x0(x, torch.arange(20).float(), scheduler, 15, args, guidance=False)
        torch.manual_seed(5)
        b = runner.estimate_x0(x, -torch.arange(20).float(), scheduler, 15, args, guidance=False)
        self.assertTrue(torch.equal(a, b))

    def test_frozen_prior_is_not_updated(self):
        args = runner.parse_args(["--runs", "1", "--log-every", "0", "--no-dynamic-mode", "frozen"])
        with patch.object(runner, "make_scheduler", wraps=runner.make_scheduler) as build:
            runner.run_variant(args, FakeAPI(), "no_dynamic_prior", 0)
            self.assertEqual(build.call_count, 1)
            self.assertIsNotNone(build.call_args[0][1])
        with patch.object(runner, "make_scheduler", wraps=runner.make_scheduler) as build:
            runner.run_variant(args, FakeAPI(), "full", 0)
            self.assertEqual(build.call_count, 1 + args.dynamic_steps)

    def test_requested_ablations_use_uniform_and_direct_x0(self):
        args = runner.parse_args(["--log-every", "0"])
        self.assertEqual(args.runs, 100)
        self.assertEqual(args.steps, 15)
        self.assertEqual(args.dynamic_steps, 10)
        self.assertEqual(args.prior_strength, 0.99)
        self.assertEqual(args.no_dynamic_mode, "uniform")
        self.assertEqual(args.no_posterior_mode, "direct_x0")
        with patch.object(runner, "make_scheduler", wraps=runner.make_scheduler) as build:
            runner.run_variant(args, FakeAPI(), "no_dynamic_prior", 0)
            self.assertEqual(build.call_count, 1)
            self.assertIsNone(build.call_args[0][1])
        x = torch.randint(0, 5, (20, 6))
        predicted_x0 = (x + 1) % 5
        with patch.object(runner, "estimate_x0", return_value=predicted_x0), \
             patch.object(runner, "d3pm_step", side_effect=AssertionError("posterior must not run")):
            actual = runner.generate(x, torch.ones(20), runner.make_scheduler(args), args.steps,
                                     "no_posterior_sampling", args)
            self.assertTrue(torch.equal(actual, predicted_x0))

    def test_no_consistency_ignores_diffusion_kernel_and_summary_is_written(self):
        args = runner.parse_args(["--log-every", "0"])
        x = torch.randint(0, 5, (20, 6))
        fitness = torch.rand(20)
        uniform = runner.make_scheduler(args)
        knowledge = runner.make_scheduler(args, [torch.zeros((6, 6), dtype=torch.long)])
        torch.manual_seed(9)
        a = runner.estimate_x0(x, fitness, uniform, 15, args, consistency=False)
        torch.manual_seed(9)
        b = runner.estimate_x0(x, fitness, knowledge, 15, args, consistency=False)
        self.assertTrue(torch.equal(a, b))
        results = [runner.run_variant(args, FakeAPI(), variant, 0) for variant in runner.VARIANTS]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            table = runner.summarize(results, output)
            self.assertTrue((output / "summary.csv").is_file())
            self.assertEqual((output / "table6.md").read_text(), table + "\n")
            for flags in runner.VARIANTS.values():
                self.assertIn(flags[0], table)

    def test_posterior_mode_and_mutation(self):
        args = runner.parse_args([])
        x = torch.randint(0, 5, (20, 6))
        scheduler = runner.make_scheduler(args, [x[:6]])
        # At t=1, a positive likelihood gives a point mass at x0.
        x0 = torch.randint(0, 5, x.shape)
        self.assertTrue(torch.equal(runner.posterior_argmax(x, x0, scheduler, 1), x0))
        args.mutation_rate = 1
        changed = runner.generate(x, None, None, 1, "random_mutation", args)
        self.assertTrue(bool((changed != x).all()))
        self.assertTrue(bool(((changed >= 0) & (changed < 5)).all()))


if __name__ == "__main__":
    unittest.main()
