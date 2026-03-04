# Permutation-based probe for attention head specialization.
# Adapted from: "Decoupling Positional and Symbolic Attention Behavior in Transformers"
# (Urrutia et al., 2024) — methodology ported to graph transformers.

import torch
import torch.nn.functional as F
from tqdm import tqdm


class HeadProbeAnalyzer:

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    def _clone_batch(self, batch):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _get_padding_mask(self, batch):
        # x[:,:,0] == 0 means padded (collator does x = x + 1)
        return batch['x'][:, :, 0].eq(0)

    def _build_perms(self, padding_mask):
        perms = []
        for b in range(padding_mask.shape[0]):
            valid = torch.where(~padding_mask[b])[0]
            if valid.numel() == 0:
                perms.append((valid, valid))
                continue
            perms.append((valid, valid[torch.randperm(valid.numel(), device=valid.device)]))
        return perms

    def permute_node_features(self, batch, perms, permute_degrees=False):
        b_ = self._clone_batch(batch)
        for b, (vi, pi) in enumerate(perms):
            if vi.numel() == 0:
                continue
            b_['x'][b, vi, :] = batch['x'][b, pi, :]
            if permute_degrees:
                b_['in_degree'][b, vi] = batch['in_degree'][b, pi]
                b_['out_degree'][b, vi] = batch['out_degree'][b, pi]
        return b_

    def permute_structure(self, batch, perms):
        b_ = self._clone_batch(batch)
        for b, (vi, pi) in enumerate(perms):
            if vi.numel() == 0:
                continue
            b_['spatial_pos'][b][vi[:, None], vi[None, :]] = batch['spatial_pos'][b][pi[:, None], pi[None, :]]
            b_['edge_input'][b][vi[:, None], vi[None, :], :, :] = batch['edge_input'][b][pi[:, None], pi[None, :], :, :]
            b_['attn_edge_type'][b][vi[:, None], vi[None, :], :] = batch['attn_edge_type'][b][pi[:, None], pi[None, :], :]
            if 'attn_bias' in batch:
                vp, pp = vi + 1, pi + 1
                b_['attn_bias'][b][vp[:, None], vp[None, :]] = batch['attn_bias'][b][pp[:, None], pp[None, :]]
                b_['attn_bias'][b][0, vp] = batch['attn_bias'][b][0, pp]
                b_['attn_bias'][b][vp, 0] = batch['attn_bias'][b][pp, 0]
        return b_

    def _cls_sim(self, a1, a2, padding_mask):
        # cosine sim on CLS row only
        valid = torch.cat([torch.ones(a1.shape[0], 1, device=padding_mask.device, dtype=torch.bool),
                           ~padding_mask], dim=1)
        c1, c2 = a1[:, 0, :][valid], a2[:, 0, :][valid]
        if c1.numel() == 0:
            return 0.0
        return F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).item()

    def _node_sim(self, a1, a2, padding_mask):
        # cosine sim over node x node submatrix (CLS excluded)
        B, T, _ = a1.shape
        full_mask = torch.cat([torch.zeros(B, 1, device=padding_mask.device, dtype=torch.bool),
                               padding_mask], dim=1)
        v2d = (~full_mask).unsqueeze(-1) & (~full_mask).unsqueeze(-2)
        v2d[:, 0, :] = False
        v2d[:, :, 0] = False
        f1, f2 = a1[v2d], a2[v2d]
        if f1.numel() == 0:
            return 0.0
        return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()

    def _bootstrap_ci(self, samples, n_boot=200, alpha=0.05, fallback_shape=None):
        if samples.numel() == 0:
            z = torch.zeros(fallback_shape) if fallback_shape else torch.zeros(0)
            return z, z
        gen = torch.Generator(device=samples.device)
        gen.manual_seed(0)
        idx = torch.randint(0, samples.shape[0], (n_boot, samples.shape[0]), generator=gen, device=samples.device)
        boots = samples[idx].mean(dim=1)
        return torch.quantile(boots, alpha / 2, dim=0).cpu(), torch.quantile(boots, 1 - alpha / 2, dim=0).cpu()

    def analyze(self, dataloader, n_permutations=10, max_batches=None, verbose=True,
                degree_policy='structure', node_node=False, bootstrap_n=200, bootstrap_alpha=0.05):
        all_pos, all_sem = [], []
        all_pos_s, all_sem_s = [], []
        all_pos_nn, all_sem_nn = [], []

        it = tqdm(dataloader, desc="Probing") if verbose else dataloader

        for i, batch in enumerate(it):
            if max_batches is not None and i >= max_batches:
                break

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if 'spatial_pos' in batch:
                cap = self.model.encoder.graph_encoder.graph_attn_bias.spatial_pos_encoder.num_embeddings - 1
                batch['spatial_pos'] = batch['spatial_pos'].clamp(max=cap)

            with torch.no_grad():
                _, orig_list = self.model(batch, return_attn=True)
            orig = torch.stack(orig_list)  # [L, H, B, T, T]
            L, H, B, T, _ = orig.shape
            pmask = self._get_padding_mask(batch)
            perm_deg = degree_policy == 'feature'

            pos_acc = torch.zeros(L, H, device=self.device)
            sem_acc = torch.zeros(L, H, device=self.device)
            pos_nn_acc = torch.zeros(L, H, device=self.device)
            sem_nn_acc = torch.zeros(L, H, device=self.device)
            pos_samps, sem_samps = [], []

            for _ in range(n_permutations):
                perms = self._build_perms(pmask)

                fb = self.permute_node_features(batch, perms, permute_degrees=perm_deg)
                with torch.no_grad():
                    _, fl = self.model(fb, return_attn=True)
                fa = torch.stack(fl)

                sb = self.permute_structure(batch, perms)
                with torch.no_grad():
                    _, sl = self.model(sb, return_attn=True)
                sa = torch.stack(sl)

                ps = torch.zeros(L, H, device=self.device)
                ss = torch.zeros(L, H, device=self.device)
                for l in range(L):
                    for h in range(H):
                        ps[l, h] = self._cls_sim(orig[l, h], fa[l, h], pmask)
                        ss[l, h] = self._cls_sim(orig[l, h], sa[l, h], pmask)
                        if node_node:
                            pos_nn_acc[l, h] += self._node_sim(orig[l, h], fa[l, h], pmask)
                            sem_nn_acc[l, h] += self._node_sim(orig[l, h], sa[l, h], pmask)

                pos_acc += ps
                sem_acc += ss
                pos_samps.append(ps.detach().cpu())
                sem_samps.append(ss.detach().cpu())

            all_pos.append((pos_acc / n_permutations).cpu())
            all_sem.append((sem_acc / n_permutations).cpu())
            if pos_samps:
                all_pos_s.append(torch.stack(pos_samps))
                all_sem_s.append(torch.stack(sem_samps))
            if node_node:
                all_pos_nn.append((pos_nn_acc / n_permutations).cpu())
                all_sem_nn.append((sem_nn_acc / n_permutations).cpu())

        pos_t = torch.stack(all_pos)
        sem_t = torch.stack(all_sem)
        ps = torch.cat(all_pos_s, dim=0) if all_pos_s else torch.empty(0)
        ss = torch.cat(all_sem_s, dim=0) if all_sem_s else torch.empty(0)
        shape = tuple(pos_t.mean(dim=0).shape)

        out = {
            'positional_mean':    pos_t.mean(dim=0),
            'semantic_mean':      sem_t.mean(dim=0),
            'positional_ci_low':  self._bootstrap_ci(ps, bootstrap_n, bootstrap_alpha, shape)[0],
            'positional_ci_high': self._bootstrap_ci(ps, bootstrap_n, bootstrap_alpha, shape)[1],
            'semantic_ci_low':    self._bootstrap_ci(ss, bootstrap_n, bootstrap_alpha, shape)[0],
            'semantic_ci_high':   self._bootstrap_ci(ss, bootstrap_n, bootstrap_alpha, shape)[1],
            'n_batches':          len(all_pos),
            'degree_policy':      degree_policy,
        }
        if node_node:
            out['positional_full_nocls_mean'] = torch.stack(all_pos_nn).mean(dim=0)
            out['semantic_full_nocls_mean']   = torch.stack(all_sem_nn).mean(dim=0)

        return out
