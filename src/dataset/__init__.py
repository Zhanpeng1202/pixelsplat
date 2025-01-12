from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_single import DatasetSingleCfg, DatasetSingle
from .dataset_4d import Dataset4dCfg, Dataset4d
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "single": DatasetSingle,
    "davis4d": Dataset4d
}


DatasetCfg = (
    DatasetRE10kCfg|
    DatasetSingleCfg|
    Dataset4dCfg
)



def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    print(f"--------{cfg.name}--------")
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)
