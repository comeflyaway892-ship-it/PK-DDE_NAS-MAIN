import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config.config import nb301_vocab_sizes
from darts_retrain.data import Cutout, cifar10_transforms, imagenet_transforms
from darts_retrain.model import NetworkCIFAR, NetworkImageNet
from darts_retrain.protocol import (
    CIFAR10_PROTOCOL,
    IMAGENET_GRADIENT_ACCUMULATION_STEPS,
    IMAGENET_PHYSICAL_BATCH_SIZE,
    IMAGENET_PROTOCOL,
)
from darts_retrain.training import (
    AdaptiveWorkerController,
    CrossEntropyLabelSmooth,
    _save_checkpoint,
    build_scheduler,
    count_parameters,
)
from evo_diff import evo_diff, sample_valid_architectures
from pipeline import (
    DEFAULT_CIFAR_DATA,
    inspect_cifar10,
    inspect_imagenet,
    retrain_stage,
    search_stage,
)
from utils.seeds import parse_seed_list
from utils.NB301 import (
    NB301Surrogate,
    genotype_to_dict,
    genotype_to_tokens,
    tokens_to_genotype,
)


class DeterministicMockSurrogate:
    def predict_tokens(self, tokens):
        values = tokens.detach().cpu().long()
        weights = torch.arange(1, values.numel() + 1)
        return 90.0 + float((values * weights).sum().item() % 400) / 100.0


class EncodingTests(unittest.TestCase):
    def test_explicit_pair_and_operation_mapping(self):
        cell = [0, 1, 1, 2, 3, 4, 4, 5, 8, 6, 0]
        tokens = torch.tensor(cell + cell)
        genotype = tokens_to_genotype(tokens)

        self.assertEqual(genotype.normal[0], ("avg_pool_3x3", 0))
        self.assertEqual(genotype.normal[1], ("max_pool_3x3", 1))
        self.assertEqual(genotype.normal[2:4], [("skip_connect", 0), ("sep_conv_3x3", 2)])
        self.assertEqual(genotype.normal[4:6], [("sep_conv_5x5", 1), ("dil_conv_3x3", 3)])
        self.assertEqual(genotype.normal[6:8], [("dil_conv_5x5", 2), ("avg_pool_3x3", 4)])
        self.assertTrue(torch.equal(genotype_to_tokens(genotype), tokens))

    def test_surrogate_receives_complete_genotype(self):
        class FakeOfficialModel:
            def __init__(self):
                self.call = None

            def predict(self, **kwargs):
                self.call = kwargs
                return 93.25

        model = FakeOfficialModel()
        surrogate = NB301Surrogate(model, with_noise=False)
        tokens = torch.zeros(22, dtype=torch.long)
        prediction = surrogate.predict_tokens(tokens)

        self.assertEqual(prediction, 93.25)
        self.assertEqual(model.call["representation"], "genotype")
        self.assertFalse(model.call["with_noise"])
        self.assertEqual(len(model.call["config"].normal), 8)
        self.assertEqual(len(model.call["config"].reduce), 8)

    def test_genotype_encoding_is_lossless(self):
        tokens = torch.zeros(22, dtype=torch.long)
        genotype = tokens_to_genotype(tokens)
        encoded = genotype_to_dict(genotype)
        self.assertEqual(genotype_to_tokens(genotype).tolist(), tokens.tolist())
        self.assertEqual(encoded["normal_concat"], [2, 3, 4, 5])
        self.assertEqual(encoded["reduce_concat"], [2, 3, 4, 5])

class DiffusionTests(unittest.TestCase):
    def assert_valid(self, architectures):
        self.assertEqual(architectures.shape[1], 22)
        for position, vocab_size in enumerate(nb301_vocab_sizes):
            self.assertTrue(torch.all(architectures[:, position] >= 0))
            self.assertTrue(torch.all(architectures[:, position] < vocab_size))

    def test_initialization_respects_each_vocabulary(self):
        self.assert_valid(sample_valid_architectures(256))

    def test_small_surrogate_search(self):
        torch.manual_seed(7)
        result = evo_diff(
            predictor=DeterministicMockSurrogate(),
            num_step=4,
            population_num=10,
            seed=7,
            plot_results=False,
            save_dir=str(MODULE_DIR / "results"),
            d3pm_eps=1e-5,
            d3pm_schedule="cosine",
            d3pm_cosine_s=0.008,
            predictor_estimator_eps=1e-12,
            predictor_temperature=0.4,
            predictor_sigma=1.0,
            select_elite_frac=0.3,
            select_eps=1e-20,
            select_fill_uniform=True,
        )
        max_accuracy, duration, uniqueness, architectures = result
        self.assertGreaterEqual(max_accuracy, 90.0)
        self.assertGreaterEqual(duration, 0.0)
        self.assertTrue(0.0 < uniqueness <= 1.0)
        self.assert_valid(architectures)


class OfficialDARTSRetrainTests(unittest.TestCase):
    def setUp(self):
        # A valid architecture using separable convolutions on every edge.
        cell = [3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3]
        self.tokens = torch.tensor(cell + cell)
        self.genotype = tokens_to_genotype(self.tokens)

    def test_locked_official_protocols(self):
        self.assertEqual(CIFAR10_PROTOCOL.epochs, 600)
        self.assertEqual(CIFAR10_PROTOCOL.batch_size, 96)
        self.assertEqual(CIFAR10_PROTOCOL.init_channels, 36)
        self.assertEqual(CIFAR10_PROTOCOL.layers, 20)
        self.assertEqual(CIFAR10_PROTOCOL.cutout_length, 16)
        self.assertEqual(CIFAR10_PROTOCOL.drop_path_prob, 0.2)
        self.assertEqual(IMAGENET_PROTOCOL.epochs, 250)
        self.assertEqual(IMAGENET_PROTOCOL.batch_size, 1024)
        self.assertEqual(IMAGENET_PROTOCOL.learning_rate, 0.5)
        self.assertEqual(IMAGENET_PROTOCOL.init_channels, 48)
        self.assertEqual(IMAGENET_PROTOCOL.layers, 14)
        self.assertEqual(IMAGENET_PROTOCOL.scheduler, "linear_warmup_to_zero")
        self.assertEqual(IMAGENET_PROTOCOL.warmup_epochs, 5)
        self.assertEqual(IMAGENET_PROTOCOL.label_smoothing, 0.1)
        self.assertEqual(IMAGENET_PHYSICAL_BATCH_SIZE, 512)
        self.assertEqual(IMAGENET_GRADIENT_ACCUMULATION_STEPS, 2)
        self.assertEqual(
            IMAGENET_PHYSICAL_BATCH_SIZE * IMAGENET_GRADIENT_ACCUMULATION_STEPS,
            IMAGENET_PROTOCOL.batch_size,
        )

    def test_official_data_augmentation_structure(self):
        cifar_train, cifar_valid = cifar10_transforms(16)
        self.assertEqual(len(cifar_train.transforms), 5)
        self.assertIsInstance(cifar_train.transforms[-1], Cutout)
        self.assertEqual(len(cifar_valid.transforms), 2)
        imagenet_train, imagenet_valid = imagenet_transforms()
        self.assertEqual(len(imagenet_train.transforms), 5)
        self.assertEqual(len(imagenet_valid.transforms), 4)

    def test_models_accept_22_token_genotype(self):
        cifar_model = NetworkCIFAR(2, 10, 3, True, self.genotype)
        cifar_model.train()
        cifar_logits, cifar_aux = cifar_model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(cifar_logits.shape), (2, 10))
        self.assertEqual(tuple(cifar_aux.shape), (2, 10))

        imagenet_model = NetworkImageNet(4, 1000, 3, True, self.genotype)
        imagenet_model.train()
        imagenet_logits, imagenet_aux = imagenet_model(
            torch.randn(2, 3, 224, 224)
        )
        self.assertEqual(tuple(imagenet_logits.shape), (2, 1000))
        self.assertEqual(tuple(imagenet_aux.shape), (2, 1000))

    def test_scheduler_and_label_smoothing(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=CIFAR10_PROTOCOL.learning_rate)
        scheduler = build_scheduler(CIFAR10_PROTOCOL, optimizer)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.025)
        optimizer.step()
        scheduler.step()
        self.assertLess(optimizer.param_groups[0]["lr"], 0.025)

        criterion = CrossEntropyLabelSmooth(3, 0.1)
        loss = criterion(torch.tensor([[2.0, 0.0, -1.0]]), torch.tensor([0]))
        self.assertTrue(torch.isfinite(loss))

    def test_adaptive_image_worker_policy(self):
        controller = AdaptiveWorkerController(4)
        self.assertEqual(controller.register_failure(), 4)
        self.assertEqual(controller.register_failure(), 2)
        for _ in range(5):
            workers = controller.register_successful_epoch()
        self.assertEqual(workers, 4)
        self.assertGreaterEqual(workers, 1)

    def test_official_darts_v2_reported_model_scale(self):
        # DARTS-V2 with each node's edges reordered into canonical predecessor
        # order. The official reported scales are 3.3M and 4.7M parameters.
        tokens = torch.tensor(
            [
                3, 3, 0, 3, 3, 0, 2, 3, 1, 2, 5,
                0, 0, 2, 0, 2, 1, 0, 2, 4, 0, 2,
            ]
        )
        genotype = tokens_to_genotype(tokens)
        cifar_model = NetworkCIFAR(36, 10, 20, True, genotype)
        imagenet_model = NetworkImageNet(48, 1000, 14, True, genotype)
        self.assertEqual(count_parameters(cifar_model, True), 3349342)
        self.assertEqual(count_parameters(imagenet_model, True), 4718752)


class PipelineTests(unittest.TestCase):
    def test_explicit_english_comma_seed_list(self):
        self.assertEqual(parse_seed_list("0,1,2,9"), [0, 1, 2, 9])
        with self.assertRaises(Exception):
            parse_seed_list("0，1，2")
        with self.assertRaises(Exception):
            parse_seed_list("0,1,1")
        with self.assertRaises(Exception):
            parse_seed_list("0,,2")

    def test_dataset_inspection(self):
        cifar = inspect_cifar10(DEFAULT_CIFAR_DATA)
        self.assertTrue(cifar["ready"])
        with tempfile.TemporaryDirectory() as directory:
            imagenet = inspect_imagenet(directory)
            self.assertFalse(imagenet["ready"])
            self.assertEqual(imagenet["train_class_directories"], 0)

    def test_retrain_saves_architecture_and_skips_imagenet_by_default(self):
        cell = [3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3]
        records = [
            {"search_seed": seed, "tokens": cell + cell}
            for seed in (0, 1)
        ]
        manifest = {"seeds": [0, 1], "architectures": records}
        args = Namespace(
            seeds=[0, 1],
            cifar_data="/tmp/cifar",
            imagenet_data="/tmp/imagenet",
            cifar_workers=0,
            imagenet_workers=0,
            gpu=0,
            parallel=False,
            train_imagenet=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "search/top1_architectures.json"
            with mock.patch(
                "pipeline.train_official_protocol", return_value=75.0
            ) as mocked_train:
                retrain_stage(args, root, manifest_path, manifest)
            self.assertTrue(
                (root / "search/architectures/search_seed_0.json").is_file()
            )
            self.assertTrue(
                (root / "search/architectures/search_seed_1.json").is_file()
            )

        calls = mocked_train.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [call[1]["protocol"].dataset for call in calls],
            ["cifar10", "cifar10"],
        )
        self.assertEqual(
            [call[1]["seed"] for call in calls], [0, 1]
        )

    def test_missing_manifest_seeds_are_searched_and_merged(self):
        cell = [3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3]
        prior_kwargs = {
            "tau": 1.0,
            "shuffle_seed": 0,
            "topk": 10,
            "lumda": 1.0,
            "dynamic_steps": 0,
        }
        args = Namespace(
            seeds=[4, 5],
            rerun_search=False,
            prior_type="uniform",
            prior_tau=1.0,
            shuffle_seed=0,
            topk=10,
            lumda=1.0,
            dynamic_steps=0,
            surrogate_path="/tmp/fake-surrogate",
            with_noise=False,
            num_step=1,
            population_num=2,
            plot=False,
        )
        manifest = {
            "seeds": [0, 1],
            "prior_type": "uniform",
            "prior_kwargs": prior_kwargs,
            "architectures": [{"search_seed": 0, "tokens": cell + cell}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "search/top1_architectures.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "pipeline.load_nb301_surrogate", return_value=object()
            ), mock.patch(
                "main.run_searches",
                return_value=[
                    {"search_seed": 4, "tokens": cell + cell},
                    {"search_seed": 5, "tokens": cell + cell},
                ],
            ) as mocked_search:
                updated = search_stage(args, root, manifest_path)

            self.assertEqual(mocked_search.call_args.kwargs["seeds"], [4, 5])
            self.assertEqual(updated["seeds"], [0, 4, 5])
            self.assertEqual(
                [item["search_seed"] for item in updated["architectures"]],
                [0, 4, 5],
            )
            self.assertTrue(
                (root / "search/architectures/search_seed_4.json").is_file()
            )

    def test_checkpoint_replacement_keeps_only_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "best.pt").write_bytes(b"stale")
            _save_checkpoint({"epoch": 1}, output)
            self.assertTrue((output / "last.pt").is_file())
            self.assertFalse((output / "best.pt").exists())
            self.assertFalse((output / "last.pt.tmp").exists())
            _save_checkpoint({"epoch": 2}, output)
            self.assertEqual(torch.load(str(output / "last.pt"))["epoch"], 2)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["last.pt"])


if __name__ == "__main__":
    unittest.main()
