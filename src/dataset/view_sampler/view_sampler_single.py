import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from ...global_cfg import get_cfg
from ...misc.step_tracker import StepTracker
from ..types import Stage
from .three_view_hack import add_third_context_index
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerSingleCfg:
    name: Literal["single"]
    index_path: Path
    num_context_views: int


class ViewSamplerSingle(ViewSampler[ViewSamplerSingleCfg]):
    # index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerSingleCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

        dacite_config = Config(cast=[tuple])
        # with cfg.index_path.open("r") as f:
        #     self.index = {
        #         k: None if v is None else from_dict(IndexEntry, v, dacite_config)
        #         for k, v in json.load(f).items()
        #     }

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        # entry = self.index.get(scene)
        # if entry is None:
        #     raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor([0,0], dtype=torch.int64, device=device)
        target_indices = torch.tensor([0,0,0], dtype=torch.int64, device=device)

        return context_indices, target_indices

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
