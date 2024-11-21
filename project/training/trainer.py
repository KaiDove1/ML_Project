import datetime
import json
from pathlib import Path
from typing import List, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from project.data.dirs import DATA_PATH
from project.data.load_patches import load_patches
from project.training.models import AttentionPooler, MaxPooler


# TO CONSIDER: Instead of training from scratch, we may want to try using an autoencoder.
def unsupervised_pretraining():
    pass


def train():
    patches_per_county = load_patches()

    results_dir = (
        Path(__file__).parent.parent.parent
        / "out"
        / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    df = pd.read_csv(DATA_PATH / "va_crops_2022.csv", index_col="county")

    # Train a CNN. For this, we will just take the mean-squared-error of each crop.
    # We will output two primary features:
    # (a) whether a crop is farmed in a county, and
    # (b) how many <units> of that crop is farmed there.
    # prediction_crops = ["barley_for_grain"]
    # prediction_crops = [c["slug"] for c in crop_ontology]

    supported_crops = []
    for crop_slug in df.columns:
        support = ~df[crop_slug].isna().sum()
        if support > 4:
            supported_crops.append(crop_slug)

    for mode in ["attention", "max_pooler"]:
        for i, crop in enumerate(supported_crops):
            print("Training model for crop", crop, f"({i+1}/{len(supported_crops)})")
            _train(
                df,
                results_dir / "single_models" / crop,
                [crop],
                patches_per_county,
                mode=mode,
            )

        print("Training combined model...")
        _train(
            df,
            results_dir / "combined_model",
            supported_crops,
            patches_per_county,
            mode=mode,
        )


def _train(
    df: pd.DataFrame,
    results_dir: Path,
    prediction_crops: List[str],
    patches_per_county: dict,
    mode="attention",
):
    per_crop_results_dir = results_dir / "per_crop"
    per_crop_results_dir.mkdir(parents=True, exist_ok=True)

    if mode == "attention":
        model = AttentionPooler(512, 512, len(prediction_crops) * 2)
    elif mode == "max_pooler":
        model = MaxPooler(512, 512, len(prediction_crops) * 2)
    else:
        raise ValueError("unsupported model type.")
    optim = torch.optim.Adam(model.parameters())

    metrics = []
    train_metrics = []

    # calculate class weights.
    class_weights = []
    for crop_slug in prediction_crops:
        class_weights.append(1 / (1 + df[crop_slug].isna().mean()))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    crop_indexes_in_df = [list(df.columns).index(c) for c in prediction_crops]

    epochs = 100
    for epoch in range(epochs):
        regression_trues = []
        regression_predictions = []
        presence_trues = []
        presence_prediction_logits = []

        for i, (county, patches) in enumerate(patches_per_county.items()):
            # Get crop counts for this county.

            if county not in df.index:
                continue

            patch_features = torch.stack(
                [patch["resnet_features"] for patch in patches]
            )
            outputs = model(patch_features)
            productivities = torch.tensor(
                df.loc[county].to_numpy()[crop_indexes_in_df],
                device=outputs.device,
                dtype=torch.float32,
            )
            has_crop = ~torch.isnan(productivities)

            presence_prediction_logits_, regression_predictions_ = torch.split(
                outputs, len(prediction_crops), dim=-1
            )

            presence_loss = (
                torch.binary_cross_entropy_with_logits(
                    presence_prediction_logits_, has_crop.float()
                )
                @ class_weights
            ).mean()
            regression_loss = (
                ((regression_predictions_[has_crop] - productivities[has_crop]) ** 2)
                @ class_weights[has_crop]
            ).mean()

            presence_prediction_logits.append(presence_prediction_logits_)
            presence_trues.append(has_crop)
            regression_predictions.append(regression_predictions_)
            regression_trues.append(productivities)

            loss = presence_loss + regression_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            metrics.append(
                {
                    "epoch": epoch,
                    "epoch_step": i,
                    "presence_loss": presence_loss.item(),
                    "regression_loss": regression_loss.item(),
                    "total_loss": loss.item(),
                }
            )

        ### Calculate statistics for model performance. ###
        # 1) f1-score, 2) RMSE.
        presence_prediction_logits = (
            torch.stack(presence_prediction_logits).T.detach().cpu().numpy()
        )
        # sigmoid(0) -> 0.5, and monotonically increasing.
        presence_predictions = presence_prediction_logits >= 0
        presence_trues = torch.stack(presence_trues).T.detach().cpu().numpy()
        regression_predictions = (
            torch.stack(regression_predictions).T.detach().cpu().numpy()
        )
        regression_trues = torch.stack(regression_trues).T.detach().cpu().numpy()

        for index, crop_slug in enumerate(prediction_crops):
            if not presence_trues[index].any():
                presence_f1 = -1
                rmse_if_present = -1
            else:
                # Calculates the RMSE for counties that have it.
                presence_f1 = f1_score(
                    presence_trues[index], presence_predictions[index]
                )
                rmse_if_present = np.sqrt(
                    np.mean(
                        (
                            regression_trues[index][presence_trues[index]]
                            - regression_predictions[index][presence_trues[index]]
                        )
                        ** 2
                    )
                )

                # Update a *cumulative* epoch metrics file.
                train_metrics.append(
                    {
                        "epoch": epoch,
                        "crop_slug": crop_slug,
                        "presence_f1": presence_f1,
                        "rmse_if_present": rmse_if_present,
                        "support": sum(presence_trues[index]),
                    }
                )

        pd.DataFrame(train_metrics).to_csv(results_dir / f"train_metrics.csv")

    # Create a plot of the results per crop.
    train_metrics_df = pd.DataFrame(train_metrics)
    for crop_slug, crop_train_metrics_df in train_metrics_df.groupby("crop_slug"):
        crop_slug = cast(str, crop_slug)

        plt.rcParams["figure.figsize"] = [8, 4]
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.title("Presence Classification F1 Score for " + crop_slug)
        plt.xlabel("Training epoch")
        plt.ylabel("f1 score")
        plt.plot(crop_train_metrics_df["epoch"], crop_train_metrics_df["presence_f1"])

        plt.subplot(1, 2, 2)
        plt.title("Regression RMSE for " + crop_slug)
        plt.xlabel("Training epoch")
        plt.ylabel("RMSE")
        plt.plot(
            crop_train_metrics_df["epoch"], crop_train_metrics_df["rmse_if_present"]
        )
        plt.tight_layout()

        ### TODO: Use an evaluation set instead of a training set.
        (per_crop_results_dir / crop_slug).mkdir()
        plt.savefig(per_crop_results_dir / crop_slug / "train_metrics_timeseries.png")
        crop_train_metrics_df.to_csv(
            per_crop_results_dir / crop_slug / "train_metrics.csv"
        )

    # Save final results
    train_metrics_df[train_metrics_df["epoch"] == epochs - 1].to_csv(
        results_dir / "final_train_metrics.csv", index=False
    )


if __name__ == "__main__":
    train()
