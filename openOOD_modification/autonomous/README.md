# Autonomous Dataset Paths

These configs are scoped to the folders already present in the workspace.

## ID datasets

- BDD100K root: `d:/Research/ood_project/data/id/bdd100k/`
- BDD100K images: `d:/Research/ood_project/data/id/bdd100k/bdd100k/bdd100k/images/`
- Cityscapes root: `d:/Research/ood_project/data/id/cityscapes/`
- Cityscapes images: `d:/Research/ood_project/data/id/cityscapes/leftImg8bit/`

## Near-OOD datasets

- LostAndFound root: `d:/Research/ood_project/data/ood/near/LostAndFound/`

## Far-OOD datasets

- SMIYC train images: `d:/Research/ood_project/data/ood/far/SMIYC/train/images/`
- SMIYC test images: `d:/Research/ood_project/data/ood/far/SMIYC/test/images/`
- StreetHazards train images: `d:/Research/ood_project/data/ood/far/StreetHazards/train/images/`
- StreetHazards test images: `d:/Research/ood_project/data/ood/far/StreetHazards/test/images/`

The imglist files referenced by the YAML templates are placed under `d:/Research/ood_project/data/benchmark_imglist/autonomous/`.

Use `autonomous_ood.yml` as the combined OOD benchmark config:
- Near-OOD: LostAndFound (`test.txt`)
- Far-OOD: SMIYC (`test.txt`) and StreetHazards (`test.txt`)