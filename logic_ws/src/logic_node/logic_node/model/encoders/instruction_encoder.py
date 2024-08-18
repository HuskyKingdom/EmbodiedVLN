import gzip
import json

import torch
import torch.nn as nn
from torch import Tensor


class InstructionEncoder(nn.Module):
    def __init__(self) -> None:
        """An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                embedding_size: The dimension of each embedding vector
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: If True, return just the final state
        """
        super().__init__()

        rnn_type = "LSTM"
        rnn = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=50,
            hidden_size=128,
            bidirectional=True,
        )
        use_pretrained_embeddings = True
        sensor_uuid = "instruction"
        if sensor_uuid:
            if use_pretrained_embeddings:
                self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(),
                    freeze=not False,
                )
            else:  # each embedding initialized to sampled Gaussian
                self.embedding_layer = nn.Embedding(
                    num_embeddings=2504,
                    embedding_dim=50,
                    padding_idx=0,
                )

    @property
    def output_size(self):
        return 128 * (1 + int(True))

    def _load_embeddings(self) -> Tensor:
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged: https://bit.ly/3u3hkYg
        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        embedding_file = "/home/ros2-agv-essentials/deeplab_ws/src/logic_node/logic_node/data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz"
        with gzip.open(embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """

        
        if True:
            instruction = observations["instruction"].long()
            print(f"first {instruction.shape}")
            lengths = (instruction != 0.0).long().sum(dim=1)
            instruction = self.embedding_layer(instruction)
            print(f"first {instruction.shape}")
        else:
            instruction = observations["rxr_instruction"]

        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).cpu()

        print(f"length {lengths}")

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths, batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        rnn_type = "LSTM"
        if rnn_type == "LSTM":
            final_state = final_state[0]
        final_state_only = True
        if final_state_only:
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
                0
            ].permute(0, 2, 1)
