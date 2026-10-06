"""Fail the image build unless torch is the CPU wheel and no NVIDIA packages are installed."""
import importlib.metadata

import chronos
import torch
import transformers
from chronos import BaseChronosPipeline, Chronos2Pipeline

assert BaseChronosPipeline and Chronos2Pipeline and chronos and transformers

version = torch.__version__
assert version.endswith("+cpu"), version
assert not torch.cuda.is_available()

nvidia = [
    dist.metadata["Name"]
    for dist in importlib.metadata.distributions()
    if "nvidia" in dist.metadata["Name"].lower()
]
assert not nvidia, nvidia
