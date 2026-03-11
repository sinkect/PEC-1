from .bridge import Extruder

try:
	from .architecture import PECEngine
except ModuleNotFoundError:  # Optional for lightweight utility imports/tests.
	PECEngine = None

try:
	from .data import PECDataset, PECCollator, EntityMasker
except ModuleNotFoundError:  # Optional for lightweight utility imports/tests.
	PECDataset = None
	PECCollator = None
	EntityMasker = None

try:
	from .dataset_mixing import (
		BlendResult,
		HFDatasetAdapter,
		build_ratio_concat_dataset,
		load_default_4_4_2_blended_dataset,
		save_blend_metadata,
		save_dataset_as_jsonl,
		save_sampled_by_source_as_jsonl,
	)
except ModuleNotFoundError:  # Optional for lightweight utility imports/tests.
	BlendResult = None
	HFDatasetAdapter = None
	build_ratio_concat_dataset = None
	load_default_4_4_2_blended_dataset = None
	save_blend_metadata = None
	save_dataset_as_jsonl = None
	save_sampled_by_source_as_jsonl = None

try:
	from .losses import (
		GateL1Trainer,
		GateL1WarmupConfig,
		compute_total_loss_with_gate_l1,
		trainer_compute_loss_with_gate_l1,
	)
except ModuleNotFoundError:  # Optional for lightweight utility imports/tests.
	GateL1Trainer = None
	GateL1WarmupConfig = None
	compute_total_loss_with_gate_l1 = None
	trainer_compute_loss_with_gate_l1 = None

__all__ = [
	"Extruder",
	"PECEngine",
	"PECDataset",
	"PECCollator",
	"EntityMasker",
	"BlendResult",
	"HFDatasetAdapter",
	"build_ratio_concat_dataset",
	"load_default_4_4_2_blended_dataset",
	"save_blend_metadata",
	"save_dataset_as_jsonl",
	"save_sampled_by_source_as_jsonl",
	"GateL1Trainer",
	"GateL1WarmupConfig",
	"compute_total_loss_with_gate_l1",
	"trainer_compute_loss_with_gate_l1",
]
