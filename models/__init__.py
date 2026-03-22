import importlib


_EXPORT_MODULES = {
    "Extruder": ".bridge",
    "PECEngine": ".architecture",
    "PECDataset": ".data",
    "PECCollator": ".data",
    "EntityMasker": ".data",
    "SharedMaskProbability": ".data",
    "BlendResult": ".dataset_mixing",
    "HFDatasetAdapter": ".dataset_mixing",
    "build_ratio_concat_dataset": ".dataset_mixing",
    "load_blended_dataset": ".dataset_mixing",
    "load_default_4_4_2_blended_dataset": ".dataset_mixing",
    "load_stage1_blended_dataset": ".dataset_mixing",
    "load_stage23_blended_dataset": ".dataset_mixing",
    "save_blend_metadata": ".dataset_mixing",
    "save_dataset_as_jsonl": ".dataset_mixing",
    "save_sampled_by_source_as_jsonl": ".dataset_mixing",
    "GateL1Trainer": ".losses",
    "GateL1WarmupConfig": ".losses",
    "ProjectorRawL2Config": ".losses",
    "compute_projector_raw_l2_loss": ".losses",
    "compute_total_loss_with_gate_l1": ".losses",
    "trainer_compute_loss_with_gate_l1": ".losses",
}


__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
