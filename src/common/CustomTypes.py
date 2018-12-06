from typing import NewType, List, Tuple
import torch

TorchTensor = NewType("TorchTensor", torch.Tensor)
TaggedSentence = NewType("TaggedSentence", List[Tuple])
