#!/usr/bin/env python3
import argparse
from pathlib import Path

import common_pb2
import numpy as np
import torch
from geotransformer.utils.pointcloud import get_nearest_neighbor
from nova.datasets import dataset_factory
from nova.proto.model import model_config_pb2

from geoTransformer.GeoTransformer.experiments.roboeye.dataset import RoboeyePairDataset
from geoTransformer.GeoTransformer.experiments.roboeye.utils import (
    compose_final_transform,
    pca_alignment,
    print_entry_summary,
    visualize_coarse_alignment,
    visualize_features,
    visualize_pair,
    visualize_transformation,
)
from iris.utils import model_config_utils


def nn_stats(a, b):
    # a, b: numpy arrays (N,3) and (M,3) ; compute NN distances from a->b
    d = get_nearest_neighbor(a, b)  # returns per a point distances
    if d.size == 0:
        return {"mean": np.inf, "rmse": np.inf, "max": np.inf}
    return {
        "mean": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "max": float(d.max()),
    }


def test_compose_final_transform():
    # synthetic raw -> preproc_src (scale+rot+trans) -> coarse -> model -> preproc_ref -> raw_ref
    # raw points
    raw_src = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32).T  # (3,1)

    # build preproc_src: R_s, t_s
    angle = 0.2
    R_s = torch.tensor(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=torch.float32,
    )
    t_s = torch.tensor([0.5, -0.2, 0.1], dtype=torch.float32)

    # coarse (applied after preproc_src)
    R_c = torch.eye(3, dtype=torch.float32)
    t_c = torch.tensor([0.05, 0.02, -0.01], dtype=torch.float32)
    T_coarse = torch.eye(4, dtype=torch.float32)
    T_coarse[:3, :3] = R_c
    T_coarse[:3, 3] = t_c

    # preproc_ref
    R_r = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float32)
    t_r = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    # model transform (maps src_coarse -> ref_proc)
    R_model = torch.tensor([[0.9, -0.1, 0], [0.1, 0.95, 0], [0, 0, 1.0]], dtype=torch.float32)
    t_model = torch.tensor([0.01, -0.02, 0.0], dtype=torch.float32)
    T_model = torch.eye(4, dtype=torch.float32)
    T_model[:3, :3] = R_model
    T_model[:3, 3] = t_model

    # compute T_final using function (convert tensors to float32 inside if needed)
    T_final = compose_final_transform(
        T_model=T_model.float(),
        preproc_R_ref=R_r.float(),
        preproc_t_ref=t_r.float(),
        preproc_R_src=R_s.float(),
        preproc_t_src=t_s.float(),
        T_coarse=T_coarse.float(),
    )

    # Forward simulate chain and compare
    # raw_src -> src_proc
    src_proc = R_s @ raw_src + t_s.unsqueeze(1)
    # src_proc -> src_coarse
    src_coarse = R_c @ src_proc + t_c.unsqueeze(1)
    # model -> ref_proc
    ref_proc = R_model @ src_coarse + t_model.unsqueeze(1)
    # ref_proc -> raw_ref (invert ref preproc)
    raw_ref_expected = torch.inverse(R_r) @ (ref_proc - t_r.unsqueeze(1))

    # Now apply T_final to raw_src
    raw_ref_via_Tfinal = T_final[:3, :3] @ raw_src + T_final[:3, 3].unsqueeze(1)

    print("expected:", raw_ref_expected.squeeze().numpy())
    print("via T_final:", raw_ref_via_Tfinal.squeeze().numpy())
    print("diff:", (raw_ref_expected - raw_ref_via_Tfinal).abs().max().item())


def main(args: argparse.Namespace) -> None:
    roboeye_root = Path.home() / "repos/roboeye/datasets/"  # Fixed path
    train_dataset_id = "HST-YZ/CAD6608043632/2025-10-14-rvbust_RE_20_synthetic"
    test_dataset_id = (
        "HST-YZ/CAD6608043632/labelled_2025-09-22-15-17-30_ConsecutiveSample_ai_detection@TAb2"
    )
    val_split_ratio = 0.3

    # get dataset proto and merge all company and projects dataset
    train_dataset = dataset_factory.get_dataset(train_dataset_id, get=True, overwrite=True)
    val_dataset = dataset_factory.get_dataset(test_dataset_id, get=True, overwrite=True)
    train_dataset.sample_read_order = common_pb2.Dataset.SampleOrder.MERGE_DATASET
    val_dataset.sample_read_order = common_pb2.Dataset.SampleOrder.MERGE_DATASET

    model_cfg = model_config_pb2.ModelConfig()
    model_config_utils.set_filtered_datasets(
        model_cfg,
        train_dataset=train_dataset,
        val_datasets=[val_dataset],
        train_split=1 - val_split_ratio,
        val_split=val_split_ratio,
    )
    # test roboeye subset train dataset
    train_pair_dataset = RoboeyePairDataset(
        dataset_root=str(roboeye_root),
        subset="train",
        num_points=1024,
        voxel_size=0.0002,
        normalize=True,
        deterministic=True,
        use_augmentation=True,
        so3_augmentation=True,
        so3_curriculum_epochs=50,  # if >0, curriculum must be handled externally
        max_so3_rotation_deg=180.0,  # if curriculum active change this gradually
        translation_jitter_m=0.005,  # 5mm
        return_corr_indices=True,
        return_normals=True,
        curr_epoch=0,
        matching_radius=0.03,
        overfitting_index=None,
        dataset_proto=model_cfg.datasets,
        calib_folder=None,
        log_dir=None,
        per_sample_cache_dir=None,
        debug=True,
    )
    # confirm keys
    print_entry_summary(train_pair_dataset.data_list[0])

    # visualize ref and src point clouds
    for i in range(200):
        sample = train_pair_dataset[i]
        if i < 197:
            continue

        proc_ref_points = sample["ref_points"].cpu().numpy()
        proc_src_points = sample["src_points"].cpu().numpy()
        ref_feats = sample["ref_feats"].cpu().numpy()
        src_feats = sample["src_feats"].cpu().numpy()
        preproc_R_ref = sample["preproc_R_ref"]
        preproc_t_ref = sample["preproc_t_ref"]
        preproc_R_src = sample["preproc_R_src"]
        preproc_t_src = sample["preproc_t_src"]
        canonical_flipped = sample.get("canonical_flipped", False)

        print("---- Preproc summary ----")
        print("canonical_flipped:", canonical_flipped)

        # 3) PCA coarse alignment (processed src -> processed ref)
        T_coarse_np = pca_alignment(proc_src_points, proc_ref_points)
        if T_coarse_np is None:
            T_coarse_np = np.eye(4, dtype=np.float32)
        print("T_coarse:", T_coarse_np)
        coarse_src_points = (T_coarse_np[:3, :3] @ proc_src_points.T + T_coarse_np[:3, 3:4]).T
        print("PCA coarse -> ref stats:", nn_stats(proc_ref_points, coarse_src_points))

        # 4) Build T_final assuming identity model (we are testing preproc & PCA composition)
        # T_model = identity (4x4) for this check: model maps PCA(preproc_src) -> preprocref
        T_model = torch.from_numpy(np.eye(4, dtype=np.float32)).float()
        T_final = compose_final_transform(
            T_model=T_model,
            preproc_R_ref=preproc_R_ref,
            preproc_t_ref=preproc_t_ref,
            preproc_R_src=preproc_R_src,
            preproc_t_src=preproc_t_src,
            T_coarse=torch.from_numpy(T_coarse_np).float(),
        )
        print("T_final (orig_src -> orig_ref) using identity model:", T_final)

        visualize_pair(
            ref_points=proc_ref_points,
            src_points=proc_src_points,
            window_name=f"Sample {i} Original Point Clouds: Green=Ref, Red=Src",
        )
        visualize_features(
            points=proc_ref_points,
            features=ref_feats,
            window_name=f"Sample {i} Original Real World Ref Features",
        )
        visualize_features(
            points=proc_src_points,
            features=src_feats,
            window_name=f"Sample {i} Original Template Src Features",
        )
        visualize_transformation(
            ref_points=proc_ref_points,
            src_points=coarse_src_points,
            tfm=T_model,
            window_name=f"Sample {i} Final Processed Model Transform: Src to Ref",
        )
        visualize_coarse_alignment(
            ref_points=proc_ref_points,
            src_points=proc_src_points,
            T_coarse=T_coarse_np,
            window_name=f"Sample {i} PCA Coarse Alignment",
        )

    # visualize ref and transformed src point clouds (should be template to real-world scans)
    # confirm the ground truth transformation is correct

    val_pair_dataset = RoboeyePairDataset(
        dataset_root=str(roboeye_root),
        subset="val",
        num_points=1024,
        voxel_size=0.0002,
        normalize=True,
        deterministic=True,
        use_augmentation=False,
        so3_augmentation=False,
        so3_curriculum_epochs=50,  # if >0, curriculum must be handled externally
        max_so3_rotation_deg=180.0,  # if curriculum active change this gradually
        translation_jitter_m=0.005,  # 5mm
        return_corr_indices=True,
        return_normals=True,
        curr_epoch=0,
        matching_radius=0.03,
        overfitting_index=None,
        dataset_proto=model_cfg.datasets,
        calib_folder=None,
        log_dir=None,
        per_sample_cache_dir=None,
        debug=True,
    )
    # confirm keys
    print_entry_summary(val_pair_dataset.data_list[0])

    # visualize ref and src point clouds
    for i in range(20):
        sample = val_pair_dataset[i]
        if i < 17:
            continue
        proc_ref_points = sample["ref_points"].cpu().numpy()
        proc_src_points = sample["src_points"].cpu().numpy()
        ref_feats = sample["ref_feats"].cpu().numpy()
        src_feats = sample["src_feats"].cpu().numpy()
        preproc_R_ref = sample["preproc_R_ref"]
        preproc_t_ref = sample["preproc_t_ref"]
        preproc_R_src = sample["preproc_R_src"]
        preproc_t_src = sample["preproc_t_src"]
        canonical_flipped = sample.get("canonical_flipped", False)

        print("---- Preproc summary ----")
        print("canonical_flipped:", canonical_flipped)

        # 3) PCA coarse alignment (processed src -> processed ref)
        T_coarse_np = pca_alignment(proc_src_points, proc_ref_points)
        if T_coarse_np is None:
            T_coarse_np = np.eye(4, dtype=np.float32)
        print("T_coarse:", T_coarse_np)
        coarse_src_points = (T_coarse_np[:3, :3] @ proc_src_points.T + T_coarse_np[:3, 3:4]).T
        print("PCA coarse -> ref stats:", nn_stats(proc_ref_points, coarse_src_points))

        # 4) Build T_final assuming identity model (we are testing preproc & PCA composition)
        # T_model = identity (4x4) for this check: model maps PCA(src_pre) -> pre_ref
        T_model = torch.from_numpy(np.eye(4, dtype=np.float32)).float()
        T_final = compose_final_transform(
            T_model=T_model,
            preproc_R_ref=preproc_R_ref,
            preproc_t_ref=preproc_t_ref,
            preproc_R_src=preproc_R_src,
            preproc_t_src=preproc_t_src,
            T_coarse=torch.from_numpy(T_coarse_np).float(),
        )
        print("T_final (orig_src -> orig_ref) using identity model:", T_final)

        visualize_pair(
            ref_points=proc_ref_points,
            src_points=proc_src_points,
            window_name=f"Sample {i} Original Point Clouds: Green=Ref, Red=Src",
        )
        visualize_features(
            points=proc_ref_points,
            features=ref_feats,
            window_name=f"Sample {i} Original Real World Ref Features",
        )
        visualize_features(
            points=proc_src_points,
            features=src_feats,
            window_name=f"Sample {i} Original Template Src Features",
        )
        visualize_transformation(
            ref_points=proc_ref_points,
            src_points=coarse_src_points,
            tfm=T_model,
            window_name=f"Sample {i} Final Processed GT Transform: Src to Ref",
        )
        visualize_coarse_alignment(
            ref_points=proc_ref_points,
            src_points=proc_src_points,
            T_coarse=T_coarse_np,
            window_name=f"Sample {i} PCA Coarse Alignment",
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="~/repos/roboeye/datasets/", help="converted dataset root",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="subset to test",
    )
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--voxel-size", type=float, default=0.009)
    parser.add_argument("--visualize", action="store_true", help="Open3D visual")
    test_compose_final_transform()
    args = parser.parse_args()
    main(args)
