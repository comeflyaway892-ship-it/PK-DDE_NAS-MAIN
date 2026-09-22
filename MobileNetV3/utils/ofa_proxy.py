import os
import sys
from types import SimpleNamespace

import numpy as np
import torch


MOBILENETV3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_EXP_ROOT = os.path.join(MOBILENETV3_ROOT, "main_exp")
for path in (MOBILENETV3_ROOT, MAIN_EXP_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from all_path import PROCESSED_DATA_PATH, UNNOISE_META_PREDICTOR_CKPT_PATH
from transfer_nag_lib.MetaD2A_mobilenetV3.loader import MetaTestDataset
from transfer_nag_lib.MetaD2A_mobilenetV3.metad2a_utils import (
    decode_ofa_mbv3_str_to_igraph,
    load_graph_config,
)
from transfer_nag_lib.MetaD2A_mobilenetV3.predictor.predictor_model import (
    PredictorModel as OFAMetaD2APredictorModel,
)
from transfer_nag_lib.ofa_net import MAX_LAYER_PER_STAGE, NUM_STAGE, OPS2STR


DEPTH_TOKEN_TO_VALUE = {0: 2, 1: 3, 2: 4}
NUM_DEPTH_TOKENS = NUM_STAGE
NUM_OP_TOKENS = NUM_STAGE * MAX_LAYER_PER_STAGE
ENCODING_LENGTH = NUM_DEPTH_TOKENS + NUM_OP_TOKENS
PREDICTOR_DATABASE_NAME = "database_219152_14.0K_train.pt"


def _load_acc_stats(data_path):
    stats_path = os.path.join(data_path, "predictor", "processed", PREDICTOR_DATABASE_NAME)
    if not os.path.exists(stats_path):
        return None, None
    data = torch.load(stats_path, map_location="cpu")
    return data.get("mean"), data.get("std")


def candidate_to_ofa_model_str(candidate):
    candidate = torch.as_tensor(candidate).detach().cpu()
    if candidate.dim() == 2:
        if candidate.shape == (ENCODING_LENGTH, len(OPS2STR)):
            candidate = candidate.argmax(dim=1)
        elif candidate.shape == (NUM_STAGE * MAX_LAYER_PER_STAGE, len(OPS2STR)):
            active = candidate.sum(dim=1) > 0
            tokens = candidate.argmax(dim=1)
            depth_by_stage = active.view(NUM_STAGE, MAX_LAYER_PER_STAGE).sum(dim=1).tolist()
            parts = []
            for i, token in enumerate(tokens.tolist()):
                stage_idx = int(i / MAX_LAYER_PER_STAGE)
                depth = int(depth_by_stage[stage_idx])
                if not active[i] or depth == 0:
                    parts.append("0-0-0")
                    continue
                if token not in OPS2STR:
                    raise ValueError(f"Unsupported OFA op token {token}; expected one of {sorted(OPS2STR)}")
                parts.append(f"{depth}-{OPS2STR[token]}")
            return "_".join(parts)
        else:
            raise ValueError(f"Unsupported one-hot OFA candidate shape: {tuple(candidate.shape)}")

    if candidate.dim() != 1:
        raise ValueError(f"Unsupported OFA candidate rank: {candidate.dim()}")
    if candidate.numel() != ENCODING_LENGTH:
        raise ValueError(
            "OFA proxy expects candidates with 25 tokens: 5 depth tokens + 20 op tokens. "
            f"Got {candidate.numel()} tokens."
        )

    candidate = candidate.long()
    depth_tokens = candidate[:NUM_DEPTH_TOKENS]
    op_tokens = candidate[NUM_DEPTH_TOKENS:]
    depth_by_stage = []
    for token in depth_tokens.tolist():
        if token not in DEPTH_TOKEN_TO_VALUE:
            raise ValueError(f"Unsupported depth token {token}; expected one of {sorted(DEPTH_TOKEN_TO_VALUE)}")
        depth_by_stage.append(DEPTH_TOKEN_TO_VALUE[token])

    parts = []
    for stage_idx, depth in enumerate(depth_by_stage):
        stage_tokens = op_tokens[
            stage_idx * MAX_LAYER_PER_STAGE : (stage_idx + 1) * MAX_LAYER_PER_STAGE
        ]
        for layer_idx, token in enumerate(stage_tokens.tolist()):
            if layer_idx >= depth:
                parts.append("0-0-0")
                continue
            if token not in OPS2STR:
                raise ValueError(f"Unsupported OFA op token {token}; expected one of {sorted(OPS2STR)}")
            parts.append(f"{depth}-{OPS2STR[token]}")
    return "_".join(parts)


class OFAProxyEvaluator:
    def __init__(
        self,
        dataset,
        *,
        num_sample=20,
        nvt=27,
        hs=512,
        nz=56,
        eval_repeats=10,
        data_path=PROCESSED_DATA_PATH,
        acc_mean=None,
        acc_std=None,
        device=None,
    ):
        self.dataset = dataset
        self.num_sample = num_sample
        self.eval_repeats = eval_repeats
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.test_dataset = MetaTestDataset(data_path, dataset, num_sample)
        loaded_mean, loaded_std = _load_acc_stats(data_path)
        self.acc_mean = loaded_mean if acc_mean is None else acc_mean
        self.acc_std = loaded_std if acc_std is None else acc_std

        args = SimpleNamespace(
            graph_data_name="ofa",
            nvt=nvt,
            hs=hs,
            nz=nz,
            num_sample=num_sample,
        )
        graph_config = load_graph_config(args.graph_data_name, args.nvt, data_path)
        self.model = OFAMetaD2APredictorModel(args, graph_config)
        ckpt_path = os.path.join(UNNOISE_META_PREDICTOR_CKPT_PATH, "ckpt_max_corr.pt")
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, candidates):
        model_strs = [candidate_to_ofa_model_str(candidate) for candidate in candidates]
        num_candidates = len(model_strs)

        with torch.no_grad():
            x = torch.stack([self.test_dataset[0] for _ in range(self.eval_repeats)]).to(self.device)
            d_mu = self.model.set_encode(x)
            if d_mu.dim() == 1:
                d_mu = d_mu.unsqueeze(0)

            arch_igraphs = [decode_ofa_mbv3_str_to_igraph(model_str) for model_str in model_strs]
            repeated_graphs = [
                arch_igraph
                for arch_igraph in arch_igraphs
                for _ in range(self.eval_repeats)
            ]
            g_mu = self.model.graph_encode(repeated_graphs)
            if g_mu.dim() == 1:
                g_mu = g_mu.unsqueeze(0)

            d_mu = d_mu.repeat(num_candidates, 1)
            y_pred = self.model.predict(d_mu, g_mu).view(num_candidates, self.eval_repeats)
            pred_acc = y_pred.mean(dim=1).detach().cpu()
            if self.acc_mean is not None and self.acc_std is not None:
                pred_acc = pred_acc * 100.0 * float(self.acc_std) + float(self.acc_mean)

        return pred_acc.float(), 1.0
