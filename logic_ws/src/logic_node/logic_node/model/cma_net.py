from torch import Size, Tensor
from torch import nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from typing import Dict, Optional, Tuple
from gym import Space

from .encoders.instruction_encoder import InstructionEncoder
from .encoders import resnet_encoders 
from .encoders.rnn_state_encoder import build_rnn_state_encoder

class CMANet(nn.Module):
    """An implementation of the cross-modal attention (CMA) network in
    https://arxiv.org/abs/2004.02857
    """

    def __init__(
        self, observation_space: Space, num_actions: int
    ) -> None:

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder()

        # Init the depth encoder
        self.depth_encoder = getattr(
            resnet_encoders, "VlnResnetDepthEncoder"
        )(
            observation_space,
            output_size=128,
            checkpoint="/home/ros2-agv-essentials/deeplab_ws/src/logic_node/logic_node/data/ddppo-models/gibson-2plus-resnet50.pth",
            backbone="resnet50",
            trainable=False,
            spatial_output=True,
        )

        # Init the RGB visual encoder
        self.rgb_encoder = getattr(
            resnet_encoders, "TorchVisionResNet50"
        )(
            256,
            normalize_visual_inputs=False,
            trainable=False,
            spatial_output=True,
        )

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        hidden_size = 512
        self._hidden_size = hidden_size

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                256,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                128,
            ),
            nn.ReLU(True),
        )

        # Init the RNN state decoder
        rnn_input_size = 128
        rnn_input_size += 256
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=512,
            rnn_type="GRU",
            num_layers=1,
        )

        self._output_size = (
            512
            + 256
            + 128
            + self.instruction_encoder.output_size
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + 256,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + 128,
            1,
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, hidden_size // 2
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type="GRU",
            num_layers=1,
        )
        self._output_size = 512

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _init_layers(self) -> None:
        if self.model_config.PROGRESS_MONITOR.use:
            nn.init.kaiming_normal_(
                self.progress_monitor.weight, nonlinearity="tanh"
            )
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)

        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)

        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (
            state,
            rnn_states_out[:, 0 : self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_states[:, 0 : self.state_encoder.num_recurrent_layers],
            masks,
        )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )

        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [
                state,
                text_embedding,
                rgb_embedding,
                depth_embedding,
                prev_actions,
            ],
            dim=1,
        )
        x = self.second_state_compress(x)
        (
            x,
            rnn_states_out[:, self.state_encoder.num_recurrent_layers :],
        ) = self.second_state_encoder(
            x,
            rnn_states[:, self.state_encoder.num_recurrent_layers :],
            masks,
        )


        return x, rnn_states_out