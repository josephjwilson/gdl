# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers,
        no_cls=False, no_degree_embedding=False,
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.no_cls = no_cls
        self.no_degree_embedding = no_degree_embedding

        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        # if self.flag and perturb is not None:
        #     node_feature += perturb

        if not self.no_degree_embedding:
            node_feature = (
                node_feature
                + self.in_degree_encoder(in_degree.clamp(max=self.in_degree_encoder.num_embeddings - 1))
                + self.out_degree_encoder(out_degree.clamp(max=self.out_degree_encoder.num_embeddings - 1))
            )

        if self.no_cls:
            return node_feature  # [B, N, C] — no graph token

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
        fixed_spd_bias=False,
        alibi_spd_bias=False,
        alibi_custom_slopes=None,
        use_virtual_distance=True,
        use_spd_bias=True,
        no_cls=False,
        use_ecc_cls_bias=False,
        ecc_cls_slope=1.0,
        use_farness_cls_bias=False,
        use_ecc_alibi_cls_bias=False,
        use_raw_ecc_alibi_cls_bias=False,
        raw_ecc_alibi_slope_rate=8.0,
        alibi_spd_slope_rate=8.0,
        use_random_cls_bias=False,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.fixed_spd_bias = fixed_spd_bias
        self.use_virtual_distance = use_virtual_distance
        self.use_spd_bias = use_spd_bias
        self.no_cls = no_cls
        self.use_ecc_cls_bias = use_ecc_cls_bias
        self.ecc_cls_slope = ecc_cls_slope
        self.use_farness_cls_bias = use_farness_cls_bias
        if use_farness_cls_bias:
            ratio = 2.0 ** (-8.0 / num_heads)
            slopes = ratio ** torch.arange(1, num_heads + 1, dtype=torch.float32)
            self.register_buffer('farness_cls_slopes', slopes.view(1, num_heads, 1))

        self.use_ecc_alibi_cls_bias = use_ecc_alibi_cls_bias
        if use_ecc_alibi_cls_bias:
            ratio = 2.0 ** (-8.0 / num_heads)
            slopes = ratio ** torch.arange(1, num_heads + 1, dtype=torch.float32)
            self.register_buffer('ecc_alibi_cls_slopes', slopes.view(1, num_heads, 1))

        self.use_raw_ecc_alibi_cls_bias = use_raw_ecc_alibi_cls_bias
        if use_raw_ecc_alibi_cls_bias:
            ratio = 2.0 ** (-raw_ecc_alibi_slope_rate / num_heads)
            slopes = ratio ** torch.arange(1, num_heads + 1, dtype=torch.float32)
            self.register_buffer('raw_ecc_alibi_cls_slopes', slopes.view(1, num_heads, 1))

        self.use_random_cls_bias = use_random_cls_bias
        if use_random_cls_bias:
            ratio = 2.0 ** (-raw_ecc_alibi_slope_rate / num_heads)
            slopes = ratio ** torch.arange(1, num_heads + 1, dtype=torch.float32)
            self.register_buffer('random_cls_slopes', slopes.view(1, num_heads, 1))

        if alibi_spd_bias:
            if alibi_custom_slopes is not None:
                slopes = torch.tensor(alibi_custom_slopes, dtype=torch.float32)
            else:
                ratio = 2.0 ** (-alibi_spd_slope_rate / num_heads)
                slopes = ratio ** torch.arange(1, num_heads + 1, dtype=torch.float32)
            self.register_buffer('alibi_slopes', slopes.view(1, num_heads, 1, 1))
            self.alibi_spd_bias = alibi_spd_bias

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        # Keep Embedding regardless of fixed_spd_bias for checkpoint compat
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )

        n_graph, n_node = x.size()[:2]

        if self.no_cls:
            # Strip CLS row/col from collator's [B, N+1, N+1] -> [B, N, N]
            graph_attn_bias = attn_bias[:, 1:, 1:].clone()
            graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
                1, self.num_heads, 1, 1
            )  # [n_graph, n_head, n_node, n_node]

            # spatial pos — apply to full matrix (no [1:,1:] offset)
            if getattr(self, 'alibi_spd_bias', False):
                actual_dist = (spatial_pos.float() - 1.0).clamp(min=0)
                spatial_pos_bias = -self.alibi_slopes * actual_dist.unsqueeze(1)
            elif self.fixed_spd_bias:
                actual_dist = spatial_pos.float() - 1.0
                bias = torch.zeros_like(actual_dist)
                bias[spatial_pos == 1] = 1.0
                dist_mask = spatial_pos >= 2
                bias[dist_mask] = 1.0 / actual_dist[dist_mask]
                spatial_pos_bias = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            else:
                spatial_pos_bias = self.spatial_pos_encoder(spatial_pos.clamp(max=self.spatial_pos_encoder.num_embeddings - 1)).permute(0, 3, 1, 2)
            if self.use_spd_bias:
                graph_attn_bias = graph_attn_bias + spatial_pos_bias

            # skip virtual distance entirely — no CLS to bias toward

            # edge features — apply to full matrix (no [1:,1:] offset)
            if self.edge_type == "multi_hop":
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1
                spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
                if self.multi_hop_max_dist > 0:
                    spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                    edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
                edge_input = self.edge_encoder(edge_input).mean(-2)
                max_dist = edge_input.size(-2)
                edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                    max_dist, -1, self.num_heads
                )
                edge_input_flat = torch.bmm(
                    edge_input_flat,
                    self.edge_dis_encoder.weight.reshape(
                        -1, self.num_heads, self.num_heads
                    )[:max_dist, :, :],
                )
                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads
                ).permute(1, 2, 3, 0, 4)
                edge_input = (
                    edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
                ).permute(0, 3, 1, 2)
            else:
                edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

            graph_attn_bias = graph_attn_bias + edge_input
            graph_attn_bias = graph_attn_bias + attn_bias[:, 1:, 1:].unsqueeze(1)  # reset

            return graph_attn_bias

        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        if getattr(self, 'alibi_spd_bias', False):
            # ALiBi-style linear distance penalty: -m_h * d(i,j)
            actual_dist = (spatial_pos.float() - 1.0).clamp(min=0)
            spatial_pos_bias = -self.alibi_slopes * actual_dist.unsqueeze(1)
        elif self.fixed_spd_bias:
            # Fixed 1/SPD bias (El et al. 2502.12352)
            # spatial_pos offset by collator: 0=padding, 1=distance_0(self), 2=distance_1, ...
            actual_dist = spatial_pos.float() - 1.0
            bias = torch.zeros_like(actual_dist)
            bias[spatial_pos == 1] = 1.0  # self-loop
            dist_mask = spatial_pos >= 2
            bias[dist_mask] = 1.0 / actual_dist[dist_mask]  # 1/d decay
            # padding (spatial_pos==0) stays 0, masked by -inf anyway
            spatial_pos_bias = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos.clamp(max=self.spatial_pos_encoder.num_embeddings - 1)).permute(0, 3, 1, 2)
        if self.use_spd_bias:
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        if self.use_virtual_distance:
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # eccentricity-based CLS→node bias: penalise peripheral nodes
        if self.use_ecc_cls_bias:
            actual_dist = spatial_pos.float() - 1.0
            actual_dist[spatial_pos == 0] = 0          # padding stays 0
            ecc = actual_dist.max(dim=-1).values       # [B, N]
            ecc_penalty = (-self.ecc_cls_slope * ecc).unsqueeze(1)  # [B, 1, N]
            graph_attn_bias[:, :, 0, 1:] = graph_attn_bias[:, :, 0, 1:] + ecc_penalty

        # farness-based ALiBi CLS→node bias: per-head geometric slopes on mean distance
        if self.use_farness_cls_bias:
            actual_dist = spatial_pos.float() - 1.0
            actual_dist[spatial_pos == 0] = 0          # padding stays 0
            n_real = (spatial_pos > 0).any(dim=-1).sum(dim=-1, keepdim=True).float().clamp(min=1)  # [B, 1]
            mean_dist = actual_dist.sum(dim=-1) / n_real  # [B, N]
            farness_penalty = -self.farness_cls_slopes * mean_dist.unsqueeze(1)  # [B, H, N]
            graph_attn_bias[:, :, 0, 1:] = graph_attn_bias[:, :, 0, 1:] + farness_penalty

        # diameter-normalized eccentricity ALiBi CLS→node bias: per-head slopes on ecc/diam
        if self.use_ecc_alibi_cls_bias:
            actual_dist = spatial_pos.float() - 1.0
            actual_dist[spatial_pos == 0] = 0                                    # padding stays 0
            ecc = actual_dist.max(dim=-1).values                                 # [B, N]
            diam = ecc.max(dim=-1, keepdim=True).values.clamp(min=1)             # [B, 1]
            norm_ecc = ecc / diam                                                # [B, N] in [~0.5, 1.0]
            ecc_alibi_penalty = -self.ecc_alibi_cls_slopes * norm_ecc.unsqueeze(1)  # [B, H, N]
            graph_attn_bias[:, :, 0, 1:] = graph_attn_bias[:, :, 0, 1:] + ecc_alibi_penalty

        # unnormalized eccentricity ALiBi CLS→node bias: per-head slopes on raw ecc
        if self.use_raw_ecc_alibi_cls_bias:
            actual_dist = spatial_pos.float() - 1.0
            actual_dist[spatial_pos == 0] = 0
            ecc = actual_dist.max(dim=-1).values                                      # [B, N]
            raw_ecc_penalty = -self.raw_ecc_alibi_cls_slopes * ecc.unsqueeze(1)        # [B, H, N]
            graph_attn_bias[:, :, 0, 1:] = graph_attn_bias[:, :, 0, 1:] + raw_ecc_penalty

        # random CLS→node bias control: uniform random scores in [0, diameter], same ALiBi slopes
        if self.use_random_cls_bias:
            actual_dist = spatial_pos.float() - 1.0
            actual_dist[spatial_pos == 0] = 0
            diam = actual_dist.max(dim=-1).values.max(dim=-1, keepdim=True).values.clamp(min=1)  # [B, 1]
            n_nodes = actual_dist.size(1)
            rand_scores = torch.rand(actual_dist.size(0), n_nodes, device=actual_dist.device) * diam  # [B, N]
            # Zero out padding positions
            rand_scores[spatial_pos.sum(dim=-1) == 0] = 0
            random_penalty = -self.random_cls_slopes * rand_scores.unsqueeze(1)  # [B, H, N]
            graph_attn_bias[:, :, 0, 1:] = graph_attn_bias[:, :, 0, 1:] + random_penalty

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias
