"""Formula, protocol and budget checks using synthetic benchmark scores."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_kernel_comparison as runner
from transition_kernels import build_kernel, KERNELS, MASK
from test_component_ablation import FakeAPI


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_equations_and_stochastic_rows(self):
        x = torch.tensor([[0, 1, 2, 3, 4, 4]])
        for kind in KERNELS:
            matrix = build_kernel(kind, knowledge=[x])
            self.assertTrue(torch.isfinite(matrix).all())
            self.assertTrue((matrix >= 0).all())
            self.assertTrue(torch.allclose(matrix.sum(-1), torch.ones(matrix.shape[:-1])))
        uniform = build_kernel("uniform")
        self.assertTrue(torch.equal(uniform, torch.full((6, 5, 5), 0.2)))
        mask = build_kernel("absorbing")
        self.assertEqual(tuple(mask.shape), (6, 6, 6))
        self.assertTrue((mask[:, :, MASK] == 1).all())
        self.assertTrue((mask[:, :, :MASK] == 0).all())
        tokens = torch.arange(5).float()
        delta = tokens[:, None] - tokens[None, :]
        gau = torch.exp(-delta.square() / (2 * 1.5**2))
        dist = torch.exp(-delta.abs() / 0.7)
        self.assertTrue(torch.allclose(build_kernel("gaussian", gaussian_sigma=[1.5])[0], gau / gau.sum(-1, keepdim=True)))
        self.assertTrue(torch.allclose(build_kernel("distance", distance_tau=[0.7])[0], dist / dist.sum(-1, keepdim=True)))
        self.assertTrue(torch.equal(build_kernel("fixed_marginal", fixed_probs=[1] * 5), uniform))
        fixed = build_kernel("fixed_marginal", fixed_probs=[1, 2, 3, 4, 5])
        self.assertTrue(torch.allclose(fixed[0, 0], torch.tensor([1, 2, 3, 4, 5]) / 15))
        self.assertTrue(torch.equal(fixed[:, 0], fixed[:, 4]))

    def test_per_edge_parameters_and_invalid_inputs(self):
        matrix = build_kernel("gaussian", gaussian_sigma=[0.5, 1, 2, 3, 4, 5])
        self.assertFalse(torch.allclose(matrix[0], matrix[1]))
        rho = torch.arange(1, 31).view(6, 5).float()
        fixed = build_kernel("fixed_marginal", fixed_probs=rho)
        self.assertTrue(torch.allclose(fixed[:, 0], rho / rho.sum(-1, keepdim=True)))
        for probs in ([0] * 5, [1, 1, -1, 1, 1], [float("nan")] * 5, [1, 2]):
            with self.assertRaises(ValueError):
                build_kernel("fixed_marginal", fixed_probs=probs)
        for scales in ([0], [-1], [float("inf")], [1, 2]):
            with self.assertRaises(ValueError):
                build_kernel("gaussian", gaussian_sigma=scales)

    def test_common_schedule_and_matrix_products(self):
        args = runner.parse_args([])
        reference = runner.make_scheduler(args, "uniform")
        for kind in KERNELS:
            scheduler = runner.make_scheduler(args, kind, [torch.ones((6, 6), dtype=torch.long)])
            self.assertTrue(torch.equal(reference.beta, scheduler.beta))
            self.assertTrue(torch.equal(reference.alpha_bar, scheduler.alpha_bar))
            product = torch.eye(scheduler.K).repeat(6, 1, 1)
            for t in range(1, args.steps + 1):
                expected = (1 - scheduler.beta[t - 1]) * torch.eye(scheduler.K) + scheduler.beta[t - 1] * scheduler.prior_kernel
                self.assertTrue(torch.allclose(scheduler.Q[t - 1], expected, atol=1e-6))
                product = product @ scheduler.Q[t - 1]
                self.assertTrue(torch.allclose(scheduler.Qbar[t], product, atol=1e-6))

    def test_paired_budget_and_absorbing_degeneracy(self):
        args = runner.parse_args(["--runs", "1", "--log-every", "0"])
        initial = None
        results = []
        for kind in KERNELS:
            api = FakeAPI()
            result = runner.run_kernel(args, api, kind, 8)
            results.append(result)
            self.assertEqual(result["architecture_evaluations"], 320)
            self.assertEqual(len(result["history"]), 15)
            self.assertTrue(all(split == "x-test" for _, split in api.reads))
            self.assertEqual(result["unique_architectures"], len(api.reads))
            self.assertEqual(result["validity_pct"], 100)
            self.assertEqual(result["completed_mask_tokens"], 0)
            if initial is None:
                initial = result["initial_population"]
            self.assertEqual(result["initial_population"], initial)
            if kind == "dynamic_marginal":
                self.assertEqual([v["after_step"] for v in result["prior_updates"]], list(range(1, 11)))
            else:
                self.assertEqual(result["prior_updates"], [])
                self.assertEqual(result["initial_kernel"], result["final_kernel"])
            if kind == "absorbing":
                self.assertEqual(result["test_accuracy"], result["initial_best_test_accuracy"])
                self.assertLessEqual(result["unique_architectures"], args.population)
        with tempfile.TemporaryDirectory() as output:
            runner.summarize(results, Path(output), args)
            table = (Path(output) / "table7.md").read_text()
            self.assertIn("degenerate", table)
            for label, _ in KERNELS.values():
                self.assertIn(label, table)

    def test_dynamic_and_uniform_match_table6(self):
        args = runner.parse_args(["--log-every", "0"])
        baseline = runner.core.parse_args(["--log-every", "0", "--prior-strength", str(args.prior_strength)])
        for kind, variant in (("dynamic_marginal", "full"), ("uniform", "no_dynamic_prior")):
            expected = runner.core.run_variant(baseline, FakeAPI(), variant, 3)
            actual = runner.run_kernel(args, FakeAPI(), kind, 3)
            for field in ("best_arch", "best_index", "test_accuracy", "unique_architectures", "architecture_evaluations"):
                self.assertEqual(expected[field], actual[field], (kind, field))
            self.assertEqual([h["best_test_accuracy"] for h in expected["history"]],
                             [h["best_test_accuracy"] for h in actual["history"]])

    def test_mask_posterior_boundaries(self):
        args = runner.parse_args([])
        scheduler = runner.make_scheduler(args, "absorbing")
        clean = torch.tensor([[0, 1, 2, 3, 4, 0]])
        observed = torch.full_like(clean, MASK)
        # At t=1 all masked observations must decode to the clean prediction.
        reconstructed, zero_mass = runner.sample_posterior(observed, clean, scheduler, 1)
        self.assertTrue(torch.equal(clean, reconstructed))
        self.assertEqual(zero_mass, 0)
        # At any t, observed unmasked tokens stay fixed for a pure absorbing kernel.
        for t in (1, 7, 15):
            result, _ = runner.sample_posterior(clean, (clean + 1) % 5, scheduler, t)
            self.assertTrue(torch.equal(clean, result))

    def test_common_forward_protocol_handles_real_masks(self):
        args = runner.parse_args(["--proposal-mode", "forward_reverse", "--log-every", "0"])
        for kind in KERNELS:
            result = runner.run_kernel(args, FakeAPI(), kind, 8)
            self.assertEqual(result["architecture_evaluations"], 320)
            self.assertEqual(result["validity_pct"], 100)
            self.assertTrue(all(token < 5 for token in result["best_arch"]))
            if kind == "absorbing":
                self.assertGreater(result["observed_mask_tokens"], 0)
                self.assertGreater(result["completed_mask_tokens"], 0)
                self.assertLess(result["raw_validity_pct"], 100)
            else:
                self.assertEqual(result["completed_mask_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
