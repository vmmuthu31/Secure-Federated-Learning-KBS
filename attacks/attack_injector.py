"""
attack_injector.py
Injects 4 attack types into malicious FL clients.

Attacks:
  1. Data Poisoning   — flip labels to a target class
  2. Model Poisoning  — scale update by large negative factor
  3. Backdoor         — embed trigger pattern + target label
  4. Byzantine        — send random Gaussian noise as update
"""

import numpy as np
from enum import Enum
from typing import Tuple

class AttackType(str, Enum):
    NONE         = "none"
    DATA_POISON  = "data_poison"
    MODEL_POISON = "model_poison"
    BACKDOOR     = "backdoor"
    BYZANTINE    = "byzantine"


class AttackInjector:
    def __init__(self, attack_type: AttackType,
                 poison_fraction: float = 0.5,
                 target_label: int = 0,
                 scale_factor: float = -5.0,
                 trigger_value: float = 1.0,
                 seed: int = 42):
        self.attack_type      = attack_type
        self.poison_fraction  = poison_fraction   # fraction of data to corrupt
        self.target_label     = target_label      # backdoor / data poison target
        self.scale_factor     = scale_factor      # model poison multiplier
        self.trigger_value    = trigger_value     # backdoor pixel trigger value
        self.rng              = np.random.default_rng(seed)

    # ── Data poisoning: flip labels ──────────────────────────────────────────
    def data_poison(self, X: np.ndarray,
                    y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_poisoned = y.copy()
        n_poison   = int(len(y) * self.poison_fraction)
        idx        = self.rng.choice(len(y), n_poison, replace=False)
        y_poisoned[idx] = self.target_label
        return X, y_poisoned

    # ── Backdoor: add pixel trigger, flip those labels ───────────────────────
    def backdoor(self, X: np.ndarray,
                 y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_bd = X.copy().astype(float)
        y_bd = y.copy()
        n_trigger = int(len(X) * self.poison_fraction)
        idx       = self.rng.choice(len(X), n_trigger, replace=False)
        # Place trigger in last 5 features (corner pixels for flat input)
        X_bd[idx, -5:] = self.trigger_value
        y_bd[idx]      = self.target_label
        return X_bd, y_bd

    # ── Model poisoning: scale the update ───────────────────────────────────
    def model_poison(self, global_w: np.ndarray,
                     local_w: np.ndarray) -> np.ndarray:
        delta = local_w - global_w
        # Scale factor < 0 → reverses gradient direction
        return global_w + self.scale_factor * delta

    # ── Byzantine: return random noise as update ─────────────────────────────
    def byzantine(self, global_w: np.ndarray,
                  noise_scale: float = 5.0) -> np.ndarray:
        return global_w + self.rng.normal(0, noise_scale, global_w.shape)

    # ── Unified interface ────────────────────────────────────────────────────
    def inject_data(self, X: np.ndarray,
                    y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.attack_type == AttackType.DATA_POISON:
            return self.data_poison(X, y)
        if self.attack_type == AttackType.BACKDOOR:
            return self.backdoor(X, y)
        return X, y   # no data-level attack

    def inject_update(self, global_w: np.ndarray,
                      local_w: np.ndarray) -> np.ndarray:
        if self.attack_type == AttackType.MODEL_POISON:
            return self.model_poison(global_w, local_w)
        if self.attack_type == AttackType.BYZANTINE:
            return self.byzantine(global_w)
        return local_w   # no update-level attack

    def __repr__(self):
        return (f"AttackInjector(type={self.attack_type}, "
                f"poison_frac={self.poison_fraction}, "
                f"target={self.target_label}, scale={self.scale_factor})")
