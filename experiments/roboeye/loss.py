from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops.pairwise_distance import pairwise_distance
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.registration.metrics import isotropic_transform_error


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg: edict):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref_feats = output_dict["ref_feats_c"]
        src_feats = output_dict["src_feats_c"]
        gt_node_corr_indices = output_dict["gt_node_corr_indices"]
        gt_node_corr_overlaps = output_dict["gt_node_corr_overlaps"]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        # Handle edge case: no positive or negative samples
        if pos_masks.sum() == 0 or neg_masks.sum() == 0:
            return torch.tensor(
                0.1, dtype=feat_dists.dtype, device=feat_dists.device, requires_grad=True
            )

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        # Sanity check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(
                0.1, dtype=feat_dists.dtype, device=feat_dists.device, requires_grad=True
            )

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg: edict):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(
        self, output_dict: Dict[str, torch.Tensor], data_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        ref_node_corr_knn_points = output_dict["ref_node_corr_knn_points"]
        src_node_corr_knn_points = output_dict["src_node_corr_knn_points"]
        ref_node_corr_knn_masks = output_dict["ref_node_corr_knn_masks"]
        src_node_corr_knn_masks = output_dict["src_node_corr_knn_masks"]
        matching_scores = output_dict["matching_scores"]
        transform = data_dict["transform"]
        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(
            ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1)
        )
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(
            torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks
        )
        slack_col_labels = torch.logical_and(
            torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks
        )

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        # edge case: no valid labels (sparse point clouds with no correspondences)
        if labels.sum() == 0:
            # Return small positive loss instead of NaN to allow gradient flow
            return torch.tensor(
                0.1, dtype=matching_scores.dtype, device=matching_scores.device, requires_grad=True
            )

        loss = -matching_scores[labels].mean()

        # Sanity check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(
                0.1, dtype=matching_scores.dtype, device=matching_scores.device, requires_grad=True
            )

        return loss


class SymmetricAwarenessLoss(nn.Module):
    """
    Symmetric Awareness Loss using Cosine Similarity.

    Encourages the model to learn symmetric feature representations by
    minimizing the cosine distance between ref->src and src->ref features.
    This helps with bidirectional matching consistency.
    """

    def __init__(self, cfg: edict):
        super(SymmetricAwarenessLoss, self).__init__()
        # Get configuration for symmetric loss
        self.use_coarse_features = getattr(cfg.symmetric_loss, "use_coarse_features", True)
        self.use_fine_features = getattr(cfg.symmetric_loss, "use_fine_features", False)
        self.temperature = getattr(cfg.symmetric_loss, "temperature", 0.1)

    def forward(self, output_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute symmetric awareness loss using cosine similarity.

        Args:
            output_dict: Dictionary containing feature embeddings
                - ref_feats_c: Reference coarse features (N, C)
                - src_feats_c: Source coarse features (M, C)
                - ref_feats_f: Reference fine features (optional)
                - src_feats_f: Source fine features (optional)

        Returns:
            Symmetric loss value
        """
        total_loss = 0.0
        num_components = 0

        # Coarse feature symmetry
        if (
            self.use_coarse_features
            and "ref_feats_c" in output_dict
            and "src_feats_c" in output_dict
        ):
            ref_feats = output_dict["ref_feats_c"]
            src_feats = output_dict["src_feats_c"]

            # Normalize features
            ref_feats_norm = F.normalize(ref_feats, p=2, dim=-1)
            src_feats_norm = F.normalize(src_feats, p=2, dim=-1)

            # Compute cosine similarity matrices
            # Forward: ref -> src
            sim_forward = torch.matmul(ref_feats_norm, src_feats_norm.t()) / self.temperature
            # Backward: src -> ref
            sim_backward = torch.matmul(src_feats_norm, ref_feats_norm.t()) / self.temperature

            # Symmetric consistency: similarity matrix should be transpose-consistent
            # sim_forward[i,j] should match sim_backward[j,i]
            sym_loss = F.mse_loss(sim_forward, sim_backward.t())

            total_loss += sym_loss
            num_components += 1

        # Fine feature symmetry (if available and enabled)
        if self.use_fine_features and "ref_feats_f" in output_dict and "src_feats_f" in output_dict:
            ref_feats_f = output_dict["ref_feats_f"]
            src_feats_f = output_dict["src_feats_f"]

            ref_feats_f_norm = F.normalize(ref_feats_f, p=2, dim=-1)
            src_feats_f_norm = F.normalize(src_feats_f, p=2, dim=-1)

            sim_forward_f = torch.matmul(ref_feats_f_norm, src_feats_f_norm.t()) / self.temperature
            sim_backward_f = torch.matmul(src_feats_f_norm, ref_feats_f_norm.t()) / self.temperature

            sym_loss_f = F.mse_loss(sim_forward_f, sim_backward_f.t())

            total_loss += sym_loss_f
            num_components += 1

        # Handle edge case: no features available
        if num_components == 0:
            return torch.tensor(
                0.0,
                dtype=ref_feats.dtype if "ref_feats" in locals() else torch.float32,
                device=ref_feats.device if "ref_feats" in locals() else torch.device("cuda"),
                requires_grad=True,
            )

        # Average over components
        loss = total_loss / num_components

        # Sanity check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, dtype=loss.dtype, device=loss.device, requires_grad=True)

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg: edict):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)

        # Add symmetric awareness loss
        self.use_symmetric_loss = getattr(cfg.loss, "use_symmetric_loss", False)
        if self.use_symmetric_loss:
            self.symmetric_loss = SymmetricAwarenessLoss(cfg)

        # Loss weights
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss
        self.weight_symmetric_loss = getattr(cfg.loss, "weight_symmetric_loss", 0.1)

    def forward(
        self, output_dict: Dict[str, torch.Tensor], data_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        loss_dict = {
            "loss": loss,
            "c_loss": coarse_loss,
            "f_loss": fine_loss,
        }

        # Add symmetric loss if enabled
        if self.use_symmetric_loss:
            symmetric_loss = self.symmetric_loss(output_dict)
            loss = loss + self.weight_symmetric_loss * symmetric_loss
            loss_dict["loss"] = loss
            loss_dict["sym_loss"] = symmetric_loss

        return loss_dict


class Evaluator(nn.Module):
    def __init__(self, cfg: edict):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rre = cfg.eval.rre_threshold
        self.acceptance_rte = cfg.eval.rte_threshold
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref_length_c = output_dict["ref_points_c"].shape[0]
        src_length_c = output_dict["src_points_c"].shape[0]
        gt_node_corr_overlaps = output_dict["gt_node_corr_overlaps"]
        gt_node_corr_indices = output_dict["gt_node_corr_indices"]
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict["ref_node_corr_indices"]
        src_node_corr_indices = output_dict["src_node_corr_indices"]

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(
        self, output_dict: Dict[str, torch.Tensor], data_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        transform = data_dict["transform"]
        ref_corr_points = output_dict["ref_corr_points"]
        src_corr_points = output_dict["src_corr_points"]
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(
        self, output_dict: Dict[str, torch.Tensor], data_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transform = data_dict["transform"]
        est_transform = output_dict["estimated_transform"]
        src_points = output_dict["src_points"]

        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def forward(
        self, output_dict: Dict[str, torch.Tensor], data_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            "PIR": c_precision,
            "IR": f_precision,
            "RRE": rre,
            "RTE": rte,
            "RMSE": rmse,
            "RR": recall,
        }
