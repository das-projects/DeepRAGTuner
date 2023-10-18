# Adapted from https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/callbacks/fault_tolerance.py
from pathlib import Path
from typing import Any

import pytorch_lightning as pl


class ModelCheckpointMine(pl.callbacks.model_checkpoint.ModelCheckpoint):

    def __init__(self, *args, fault_tolerant=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_tolerant = fault_tolerant

    def on_exception(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
        if self.fault_tolerant:
            # overwrite if necessary
            trainer.save_checkpoint(str(Path(self.dirpath) / '.pl_auto_save.ckpt'))
