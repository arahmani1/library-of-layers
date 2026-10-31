import torch
from torch.nn import functional as F

from .lol_window import LoLWindow


class LoLDynamicWindow(LoLWindow):
    def _attention_architecture(self):
        return "lol_dynamic_window"

    def __init__(self, config):
        super().__init__(config)

        self.register_buffer(
            "expert_bias", torch.zeros(self.n_experts, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # Router-owned first windows participate in the same balancing statistics.
    # ------------------------------------------------------------------
    def _sequence_aux_loss(
        self, unbiased_indices, scores, doc_of_window, active_mask,
    ):
        """DeepSeek-V3 balance loss, averaged over eligible documents."""
        doc_ids = doc_of_window[active_mask]
        if doc_ids.numel() == 0:
            return scores.sum() * 0.0

        unique_docs, inverse = torch.unique(
            doc_ids, sorted=True, return_inverse=True
        )
        n_docs = unique_docs.numel()
        affinities = scores[active_mask].float()
        hard = F.one_hot(
            unbiased_indices[active_mask], self.n_experts
        ).float()

        hard_sum = torch.zeros(
            n_docs, self.n_experts, device=affinities.device,
            dtype=affinities.dtype,
        )
        affinity_sum = torch.zeros_like(hard_sum)
        hard_sum.index_add_(0, inverse, hard)
        affinity_sum.index_add_(0, inverse, affinities)
        doc_count = torch.bincount(
            inverse, minlength=n_docs
        ).to(affinities.dtype).unsqueeze(-1)

        frequency = hard_sum / doc_count
        mean_affinity = affinity_sum / doc_count
        return self.n_experts * (frequency * mean_affinity).sum(-1).mean()

    def _router_bias(self, depth):
        del depth
        return self.expert_bias

    def _router_statistics(
        self, scores, unbiased_indices, indices, metadata,
        return_router_diagnostics=True,
    ):
        active_mask = metadata.active_window_mask
        active = active_mask.float()
        oh = F.one_hot(indices, self.n_experts).float()
        oh_sum = (oh * active.unsqueeze(-1)).sum(dim=0)
        raw_scores = scores.float()
        p_sum = (raw_scores * active.unsqueeze(-1)).sum(dim=0)
        count = active.sum()

        denom = count.clamp_min(1.0)
        aux_term = self.n_experts * ((oh_sum / denom) * (p_sum / denom)).sum()
        sequence_aux = self._sequence_aux_loss(
            unbiased_indices, scores, metadata.doc_of_window, active_mask
        )

        # Sufficient statistics for both balancing strategies. Hard assignment
        # counts use the actual (biased) dispatch; sequence_aux uses unbiased
        # choices and affinity scores, matching DeepSeek-V3.
        stats = {
            "oh_sum": oh_sum.detach(),
            "p_sum": p_sum,
            "count": count.detach(),
            "sequence_aux_loss": sequence_aux,
        }
        if return_router_diagnostics:
            diagnostic_probabilities = raw_scores / raw_scores.sum(
                -1, keepdim=True
            ).clamp_min(1e-9)
            router_entropy_sum = (
                -(
                    diagnostic_probabilities
                    * diagnostic_probabilities.clamp_min(1e-9).log()
                ).sum(-1)
                * active
            ).sum().detach()
            router_confidence_sum = (
                diagnostic_probabilities.max(dim=-1).values * active
            ).sum().detach()
            stats.update(
                router_entropy_sum=router_entropy_sum,
                router_confidence_sum=router_confidence_sum,
            )
        return aux_term, stats

    def _format_expert_traces(self, traces):
        return traces
