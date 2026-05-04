# OpenOOD Modifications

## Added files
- `autonomous_configs/` — dataset and pipeline configuration for autonomous driving 
  OOD evaluation covering BDD100K (ID), LostAndFound (near-OOD), StreetHazards 
  and SMIYC (far-OOD). Created from scratch for this project.

## Known framework issues
- `openood/postprocessors/mds_postprocessor.py` contains a batch-size dependency 
  bug that produces degenerate Mahalanobis scores. A correct custom implementation 
  is provided in `mahalanobis_eval.py` at the project root.
