import torch.nn as nn
from torch.nn.parameter import Parameter
import torch

import math


class MoE(nn.Module):
    def __init__(self, input_dim, tasks, d_hid, number_of_experts=6, bias=False):
        super(MoE, self).__init__()
        self.input_dim = input_dim
        self.number_of_experts = number_of_experts
        self.tasks = tasks

        self.experts = Parameter(torch.Tensor(input_dim, d_hid, number_of_experts))
        if bias:
            self.bias = Parameter(torch.Tensor(input_dim))
        else:
            self.register_parameter("bias", None)

        self.gates = {}  # TODO: Typed. Task -> {}
        for task in tasks:
            self.gates[task] = Parameter(torch.Tensor(input_dim, number_of_experts))
            # self.gates[task] = nn.Linear(self.input_dim, number_of_experts)

        nn.init.kaiming_uniform_(self.experts, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.experts)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        for task in self.tasks:
            nn.init.kaiming_uniform_(self.gates[task], a=math.sqrt(5))

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim

    def forward(self, encoded_sent, task):
        """

		:param encoded_sent: [batch, length, d]
		:param task:
		:return:
		"""
        print("moe called for {}".format(task.name))
        # expert_out is [batch, length, hidden_d, nb_experts]
        expert_out = torch.tensordot(a=encoded_sent, b=self.experts, dims=1)

        # gate_out is [batch, length, 1, nb_experts]
        gate = self.gates[task]
        gate_out = torch.tensordot(a=encoded_sent, b=gate, dims=1)
        gate_out = gate_out.unsqueeze(2)

        # aggregate across experts
        # out is [batch, length, hidden_d]
        out = expert_out * gate_out
        out = out.sum(dim=-1)
        return out
