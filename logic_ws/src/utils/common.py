from torch import Size, Tensor
from torch import nn as nn
import torch
from gym import spaces
import gzip
import json

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

class CustomFixedCategorical(torch.distributions.Categorical):  # type: ignore
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1, keepdim=True)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().unsqueeze(-1)
    



class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        x = self.linear(x)
        return CustomFixedCategorical(logits=x.float(), validate_args=False)




def single_frame_box_shape(box: spaces.Box) -> spaces.Box:
    """removes the frame stack dimension of a Box space shape if it exists."""
    if len(box.shape) < 4:
        return box

    return spaces.Box(
        low=box.low.min(),
        high=box.high.max(),
        shape=box.shape[1:],
        dtype=box.high.dtype,
    )



def batch_obs(observations: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    batched_obs = {}
    for key, tensor in observations.items():
        # Add a batch dimension (bs=1) and move the tensor to the specified device
        batched_obs[key] = tensor.unsqueeze(0).to(device)
    return batched_obs



def tokenize(text: str) -> List[str]:
    return text.lower().split()

def text_to_indices(text: str, vocab: Dict[str, int]) -> List[int]:
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def text_to_tensor(text: str) -> torch.Tensor:
    vocab_file_path = "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
    with gzip.open(vocab_file_path, 'rt', encoding='utf-8') as f:
        vocab = json.load(f)
        
    # Convert text to indices
    indices = text_to_indices(text, vocab)
    
    # Since there's only one text, we don't need to pad to a max_length
    indices_tensor = torch.tensor([indices], dtype=torch.long)  # Add batch dimension
    
    return indices_tensor