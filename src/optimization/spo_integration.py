"""Smart Predict, then Optimize (SPO+) integration.

Uses PyEPO to train ML models that optimize downstream decision quality,
not just prediction accuracy.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger


def create_spo_loss(optmodel):
    """Create SPO+ loss function from a PyEPO optimization model.

    SPO+ loss measures decision quality: how much worse is the decision
    made with predicted costs vs. true costs.
    """
    from pyepo.func import SPOPlus

    spo_loss = SPOPlus(optmodel, processes=1)
    logger.info("Created SPO+ loss function")
    return spo_loss


def train_spo_model(
    model: torch.nn.Module,
    optmodel,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
) -> list[float]:
    """Train a neural net with SPO+ loss to optimize decisions directly.

    Instead of minimizing MSE(predicted_PD, true_PD), this minimizes
    the regret of portfolio decisions made with predicted PDs.
    """
    from pyepo.func import SPOPlus

    spo_loss = SPOPlus(optmodel, processes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, costs in train_loader:
            optimizer.zero_grad()
            predicted_costs = model(features)
            loss = spo_loss(predicted_costs, costs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            logger.info(f"SPO+ Epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}")

    return losses


def compare_two_stage_vs_spo(
    predictions_two_stage: np.ndarray,
    predictions_spo: np.ndarray,
    true_costs: np.ndarray,
    optmodel,
) -> dict[str, float]:
    """Compare traditional two-stage approach vs SPO+.

    Measures decision quality (portfolio return) not just prediction accuracy.
    """
    from pyepo.metric import regret

    regret_2stage = regret(predictions_two_stage, true_costs, optmodel)
    regret_spo = regret(predictions_spo, true_costs, optmodel)

    result = {
        "regret_two_stage": float(regret_2stage),
        "regret_spo": float(regret_spo),
        "improvement_pct": float((regret_2stage - regret_spo) / regret_2stage * 100),
    }
    logger.info(f"SPO+ improvement: {result['improvement_pct']:.1f}% lower regret")
    return result
