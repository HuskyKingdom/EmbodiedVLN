from torch import Size, Tensor
from torch import nn as nn
import torch
from gym import spaces
import gzip
import json
import numpy as np
import cv2

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


def downsampling(src,h,w):
    result = cv2.resize(src, (h, w), interpolation=cv2.INTER_LINEAR)
    return result



def generate_random_obs(space):
    random_obs = {}
    for key, box in space.spaces.items():
        if box.dtype == np.uint8:
            random_obs[key] = torch.randint(low=0, high=256, size=box.shape, dtype=torch.uint8)
        elif box.dtype == np.float32:
            random_obs[key] = torch.rand(size=box.shape, dtype=torch.float32)
    return random_obs

def generate_null_obs(space):
    null_obs = {}
    for key, box in space.spaces.items():
        null_obs[key] = None
    return null_obs


def batch_obs(observations: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    batched_obs = {}
    for key, tensor in observations.items():
        # Add a batch dimension (bs=1) and move the tensor to the specified device
        batched_obs[key] = tensor.unsqueeze(0).to(device)
    return batched_obs





def load_vocab(file_path: str) -> Dict[str, int]:
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
        vocab = data['instruction_vocab']['word2idx_dict']
        vocab['<UNK>'] = data['instruction_vocab']['UNK_INDEX']
        vocab['<PAD>'] = data['instruction_vocab']['PAD_INDEX']
    return vocab

def tokenize(text: str) -> List[str]:
    """Tokenizes input text into a list of words."""
    return text.lower().split()

def text_to_indices(text: str, vocab: Dict[str, int]) -> List[int]:
    """Converts text to a list of indices based on the vocabulary."""
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad_or_truncate(indices: List[int], max_length: int, pad_index: int) -> List[int]:
    """Pads or truncates the list of indices to the specified length."""
    if len(indices) > max_length:
        return indices[:max_length]
    return indices + [pad_index] * (max_length - len(indices))

def text_to_tensor(text: str, vocab: Dict[str, int], max_length: int = 200) -> torch.Tensor:
    """Converts a single text to a tensor of token indices with padding/truncating."""
    indices = text_to_indices(text, vocab)
    indices_padded = pad_or_truncate(indices, max_length, vocab['<PAD>'])
    indices_tensor = torch.tensor(indices_padded, dtype=torch.long)
    return indices_tensor


# def load_vocab(file_path: str) -> Dict[str, int]:
#     with gzip.open(file_path, 'rt', encoding='utf-8') as f:
#         data = json.load(f)
#         vocab = data['instruction_vocab']['word2idx_dict']
#         vocab['<UNK>'] = data['instruction_vocab']['UNK_INDEX']
#         vocab['<PAD>'] = data['instruction_vocab']['PAD_INDEX']
#     return vocab

# def tokenize(text: str) -> List[str]:
#     """Tokenizes input text into a list of words."""
#     return text.lower().split()

# def text_to_indices(text: str, vocab: Dict[str, int]) -> List[int]:
#     """Converts text to a list of indices based on the vocabulary."""
#     tokens = tokenize(text)
#     return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

# def text_to_tensor(text: str, vocab: Dict[str, int]) -> torch.Tensor:
#     """Converts a single text to a tensor of token indices."""
#     indices = text_to_indices(text, vocab)
#     indices_tensor = torch.tensor(indices, dtype=torch.long)  # Add batch dimension
#     return indices_tensor


# def tokenize(text: str) -> List[str]:
#     return text.lower().split()

# def text_to_indices(text: str, vocab: Dict[str, int]) -> List[int]:
#     tokens = tokenize(text)
#     return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

# def text_to_tensor(text: str) -> torch.Tensor:
#     vocab_file_path = "/home/ros2-agv-essentials/deeplab_ws/src/logic_node/logic_node/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
#     import os
#     print(os.getcwd())
#     with gzip.open(vocab_file_path, 'rt', encoding='utf-8') as f:
#         vocab = json.load(f)
        
#     # Convert text to indices
#     indices = text_to_indices(text, vocab)
    
#     # Since there's only one text, we don't need to pad to a max_length
#     indices_tensor = torch.tensor([indices], dtype=torch.long)  # Add batch dimension
    
#     return indices_tensor