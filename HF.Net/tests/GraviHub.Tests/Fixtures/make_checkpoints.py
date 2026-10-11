"""Writes the same small state dict in both torch.save layouts.

    python make_checkpoints.py

legacy.bin is the pre-torch-1.6 layout (a run of bare pickles followed by the raw storages), still
what many Hub checkpoints ship - ProsusAI/finbert among them. zip.bin is the current ZIP layout.
Every value is exactly representable in its dtype, so the C# tests can assert equality, and the
expected values are written out by hand in PyTorchCheckpointTests rather than read back from here.
"""
import os

import torch

here = os.path.dirname(os.path.abspath(__file__))

weight = torch.tensor([[1.5, -2.25, 3.125], [4.0, 5.5, -6.75]], dtype=torch.float32)
state = {
    "encoder.weight": torch.nn.Parameter(weight.clone()),
    # A transposed view: shares the weight's storage with strides (1, 3), so it is non-contiguous.
    "encoder.weight_t": weight.t(),
    "encoder.bias": torch.tensor([0.5, -0.5], dtype=torch.float32),
    "half": torch.tensor([0.25, -1.0, 2048.0], dtype=torch.float16),
    "ids": torch.tensor([3, -7, 2**40], dtype=torch.int64),
    "scalar": torch.tensor(42.0, dtype=torch.float64),
    # A slice starting part-way into its storage: exercises a non-zero storage offset.
    "tail": torch.arange(10, dtype=torch.float32)[6:],
}

torch.save(state, os.path.join(here, "legacy.bin"), _use_new_zipfile_serialization=False)
torch.save(state, os.path.join(here, "zip.bin"))
print("torch", torch.__version__)
