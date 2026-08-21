"""File for storing simulation scenarios of interest.
We also define keywords that are used in all simulations here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

# Shared keywords for repeated simulations

WELLBEING_CLIP = (0.0, 10.0)
"""Standard wellbeing clipping bounds used across scenarios."""

SMU_CLIP = (0.0, 16.0)
"""Standard SMU clipping bounds used across scenarios."""

INITIAL_AGENT_PRIORS = {
    "wb_mu": 6.0,
    "wb_sigma": 1.0,
    "smu_mu": 3.0,
    "smu_sigma": 0.2,
}
"""Gaussian priors for initial wellbeing/SMU when seeding agents."""


@dataclass
class Scenario:
    """Container describing a single ABD simulation configuration."""

    label: str
    seed: int
    horizon: int = 1000
    n_agents: int = 50
    decision_name: str = "default"
    decision_overrides: Mapping[str, float] = field(default_factory=dict)
    learning_name: str = "regression_based"
    learning_overrides: Mapping[str, float] = field(default_factory=dict)
    como_name: str = "population_mean"
    como_overrides: Mapping[str, float] = field(default_factory=dict)
    adjacency_neighbours: int = 1
    agent_parameter_overrides: Mapping[str, float] = field(default_factory=dict)


NEG_POP_POS_INDCOR = Scenario(
    label="Declining Population WB, Positive Individual Correlation",
    seed=7,
    horizon=1000,
    n_agents=100,
    decision_overrides={
        "rho": 0.7,
        "smu_wb_coeff": -0.5,
        "como_wb_coeff": -0.8,
        "delayed_decline": "simple",
        "boost_coeff": 0.5,
        "decline_coeff": 0.2,
        "decline_lag": 15,
        "wellbeing_bounds": WELLBEING_CLIP,
        "smu_bounds": SMU_CLIP,
    },
    learning_overrides={"learning_rate": 0.1},
    agent_parameter_overrides=dict(INITIAL_AGENT_PRIORS),
)

NEG_POP_COMO_DISABLED = Scenario(
    label="Declining Population WB, COMO disabled",
    seed=7,
    horizon=1000,
    n_agents=500,
    decision_overrides={
        "boost_coeff": 0.5,
        "como_wb_coeff": 0.0,
        "decline_coeff": 0.6,
        "decline_lag": 15,
        "delayed_decline": "simple",
        "rho": 0.7,
        "smu_bounds": SMU_CLIP,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
    },
    learning_overrides={"learning_rate": 0.1},
    como_name="population_mean_positive",
    agent_parameter_overrides=dict(INITIAL_AGENT_PRIORS),
)

NEG_POP_POS_INDCOR_POS_COMO = Scenario(
    label="Declining Population WB, Positive Individual Correlation, Positive COMO, simple",
    seed=7,
    horizon=1000,
    n_agents=500,
    decision_overrides={
        "rho": 0.7,
        "smu_wb_coeff": -0.5,
        "como_wb_coeff": -0.8,
        "delayed_decline": "simple",
        "boost_coeff": 0.5,
        "decline_coeff": 0.2,
        "decline_lag": 15,
        "wellbeing_bounds": WELLBEING_CLIP,
        "smu_bounds": SMU_CLIP,
    },
    learning_overrides={"learning_rate": 0.1},
    como_name="population_mean_positive",
    agent_parameter_overrides=dict(INITIAL_AGENT_PRIORS),
)

NEG_POP_POS_INDCOR_POS_COMO_SMOOTH = Scenario(
    label="Declining Population WB, Positive Individual Correlation, Positive COMO",
    seed=7,
    horizon=1000,
    n_agents=500,
    decision_overrides={
        "rho": 0.7,
        "smu_wb_coeff": -0.5,
        "como_wb_coeff": -0.8,
        "delayed_decline": "smooth",
        "reward_scale": 0.5,
        "penalty_scale": 0.05,
        "fast_timescale": 1.0,
        "slow_timescale": 15.0,
        "effect_horizon": 15,
        "wellbeing_bounds": WELLBEING_CLIP,
        "smu_bounds": SMU_CLIP,
    },
    learning_overrides=dict(NEG_POP_POS_INDCOR_POS_COMO.learning_overrides),
    como_name=NEG_POP_POS_INDCOR_POS_COMO.como_name,
    agent_parameter_overrides=dict(INITIAL_AGENT_PRIORS),
)


NEG_POP_POS_INDCOR_RING_MULTI_HOP_SMOOTH = Scenario(
    label=(
        "Declining Population WB, Positive Individual Correlation -- "
        "Ring adjacency (multi-hop), smooth decline"
    ),
    seed=NEG_POP_POS_INDCOR.seed + 30,
    horizon=NEG_POP_POS_INDCOR.horizon,
    n_agents=NEG_POP_POS_INDCOR.n_agents,
    decision_overrides={
        "rho": 0.7,
        "smu_wb_coeff": -0.5,
        "como_wb_coeff": -0.8,
        "delayed_decline": "smooth",
        "reward_scale": 1.2,
        "penalty_scale": 0.7,
        "fast_timescale": 15.0,
        "slow_timescale": 2.5,
        "effect_horizon": 30,
        "target_total_effect": 0.0,
        "wellbeing_bounds": WELLBEING_CLIP,
        "smu_bounds": SMU_CLIP,
    },
    learning_name=NEG_POP_POS_INDCOR.learning_name,
    learning_overrides=dict(NEG_POP_POS_INDCOR.learning_overrides),
    como_name="adjacency_positive",
    como_overrides={"adjacency_kind": "ring"},
    adjacency_neighbours=3,
    agent_parameter_overrides=dict(INITIAL_AGENT_PRIORS),
)


AR1_POP_MEAN_GAP_GRAD_ASCENT = Scenario(
    label="AR1 Population mean gap -- gradient ascent learning",
    seed=43,
    horizon=1000,
    n_agents=100,
    decision_name="ar1",
    decision_overrides={
        "como_wb_coeff": -0.8,
        "smu_bounds": SMU_CLIP,
        "smu_decay": 0.8,
        "smu_mean_reversion": 0.2,
        "smu_noise": 0.5,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
        "wellbeing_noise": 0.5,
    },
    learning_name="gradient_ascent",
    learning_overrides={
        "learning_rate": 0.1,
        "max_step": 0.55,
    },
    como_name="population_mean",
    adjacency_neighbours=1,
)


AR1_POP_POS_GAP_GRAD_ASCENT = Scenario(
    label="AR1 Population positive gap -- gradient ascent learning",
    seed=53,
    horizon=1000,
    n_agents=100,
    decision_name="ar1",
    decision_overrides={
        "como_wb_coeff": -0.8,
        "smu_bounds": SMU_CLIP,
        "smu_decay": 0.8,
        "smu_mean_reversion": 0.2,
        "smu_noise": 0.5,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
        "wellbeing_noise": 0.5,
    },
    learning_name="gradient_ascent",
    learning_overrides={
        "learning_rate": 0.1,
        "max_step": 0.55,
    },
    como_name="population_mean_positive",
    adjacency_neighbours=1,
)


DEFAULT_SIMPLE_DELAY_REGRESSION_NO_ADJ_POP_POS_GAP = Scenario(
    label="Default (simple delay) -- regression learning -- no adjacency -- Population positive gap",
    seed=82,
    horizon=1000,
    n_agents=100,
    decision_name="default",
    decision_overrides={
        "boost_coeff": 0.6,
        "como_wb_coeff": -0.3,
        "decline_coeff": 0.12,
        "decline_lag": 12,
        "delayed_decline": "simple",
        "rho": 0.4,
        "smu_bounds": SMU_CLIP,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
    },
    learning_name="regression_based",
    learning_overrides={"learning_rate": 0.05},
    como_name="population_mean_positive",
    adjacency_neighbours=0,
)


DEFAULT_NO_DELAY_REGRESSION_NO_ADJ_GOLDILOCKS_1_4 = Scenario(
    label="Default (no delay) -- regression learning -- no adjacency -- Goldilocks (width 1.4)",
    seed=102,
    horizon=1000,
    n_agents=100,
    decision_name="default",
    decision_overrides={
        "boost_coeff": 0.0,
        "como_wb_coeff": -0.3,
        "decline_coeff": 0.0,
        "decline_lag": 10,
        "delayed_decline": "False",
        "penalty_scale": 0.0,
        "reward_scale": 0.0,
        "rho": 0.4,
        "smu_bounds": SMU_CLIP,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
    },
    learning_name="regression_based",
    learning_overrides={"learning_rate": 0.05},
    como_name="goldilocks",
    como_overrides={"width": 1.4},
    adjacency_neighbours=0,
)


DEFAULT_SIMPLE_DELAY_REGRESSION_NO_ADJ_GOLDILOCKS_1_4 = Scenario(
    label="Default (simple delay) -- regression learning -- no adjacency -- Goldilocks (width 1.4)",
    seed=112,
    horizon=1000,
    n_agents=100,
    decision_name="default",
    decision_overrides={
        "boost_coeff": 0.6,
        "como_wb_coeff": -0.3,
        "decline_coeff": 0.12,
        "decline_lag": 12,
        "delayed_decline": "simple",
        "rho": 0.4,
        "smu_bounds": SMU_CLIP,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
    },
    learning_name="regression_based",
    learning_overrides={"learning_rate": 0.05},
    como_name="goldilocks",
    como_overrides={"width": 1.4},
    adjacency_neighbours=0,
)


DEFAULT_SIMPLE_DELAY_REGRESSION_RING_POP_MEAN_GAP = Scenario(
    label="Default (simple delay) -- regression learning -- ring (2 neighbours) -- Population mean gap",
    seed=152,
    horizon=1000,
    n_agents=100,
    decision_name="default",
    decision_overrides={
        "boost_coeff": 0.6,
        "como_wb_coeff": -0.3,
        "decline_coeff": 0.12,
        "decline_lag": 12,
        "delayed_decline": "simple",
        "rho": 0.4,
        "smu_bounds": SMU_CLIP,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
    },
    learning_name="regression_based",
    learning_overrides={"learning_rate": 0.05},
    como_name="adjacency",
    como_overrides={"adjacency_kind": "ring"},
    adjacency_neighbours=2,
)


DEFAULT_NO_DELAY_REGRESSION_RING_POP_POS_GAP = Scenario(
    label="Default (no delay) -- regression learning -- ring (2 neighbours) -- Population positive gap",
    seed=172,
    horizon=1000,
    n_agents=100,
    decision_name="default",
    decision_overrides={
        "boost_coeff": 0.0,
        "como_wb_coeff": -0.3,
        "decline_coeff": 0.0,
        "decline_lag": 10,
        "delayed_decline": "False",
        "penalty_scale": 0.0,
        "reward_scale": 0.0,
        "rho": 0.4,
        "smu_bounds": SMU_CLIP,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
    },
    learning_name="regression_based",
    learning_overrides={"learning_rate": 0.05},
    como_name="adjacency_positive",
    como_overrides={"adjacency_kind": "ring"},
    adjacency_neighbours=2,
)


DEFAULT_NO_DELAY_PEER_RING_POP_POS_GAP = Scenario(
    label="Default (no delay) -- peer learning -- ring (2 neighbours) -- Population positive gap",
    seed=174,
    horizon=1000,
    n_agents=100,
    decision_name="default",
    decision_overrides={
        "boost_coeff": 0.0,
        "como_wb_coeff": -0.3,
        "decline_coeff": 0.0,
        "decline_lag": 10,
        "delayed_decline": "False",
        "penalty_scale": 0.0,
        "reward_scale": 0.0,
        "rho": 0.4,
        "smu_bounds": SMU_CLIP,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
    },
    learning_name="peer_learning",
    learning_overrides={"learning_rate": 0.05},
    como_name="adjacency_positive",
    como_overrides={"adjacency_kind": "ring"},
    adjacency_neighbours=2,
)


DEFAULT_SIMPLE_DELAY_REGRESSION_RING_GOLDILOCKS_1_4 = Scenario(
    label="Default (simple delay) -- regression learning -- ring (2 neighbours) -- Goldilocks (width 1.4)",
    seed=212,
    horizon=1000,
    n_agents=100,
    decision_name="default",
    decision_overrides={
        "boost_coeff": 0.6,
        "como_wb_coeff": -0.3,
        "decline_coeff": 0.12,
        "decline_lag": 12,
        "delayed_decline": "simple",
        "rho": 0.4,
        "smu_bounds": SMU_CLIP,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
    },
    learning_name="regression_based",
    learning_overrides={"learning_rate": 0.05},
    como_name="adjacency_goldilocks",
    como_overrides={"adjacency_kind": "ring", "width": 1.4},
    adjacency_neighbours=2,
)


DEFAULT_SIMPLE_DELAY_REGRESSION_GROUPS_POP_MEAN_GAP = Scenario(
    label="Default (simple delay) -- regression learning -- group clusters -- Population mean gap",
    seed=252,
    horizon=1000,
    n_agents=100,
    decision_name="default",
    decision_overrides={
        "boost_coeff": 0.6,
        "como_wb_coeff": -0.3,
        "decline_coeff": 0.12,
        "decline_lag": 12,
        "delayed_decline": "simple",
        "rho": 0.4,
        "smu_bounds": SMU_CLIP,
        "smu_wb_coeff": -0.5,
        "wellbeing_bounds": WELLBEING_CLIP,
    },
    learning_name="regression_based",
    learning_overrides={"learning_rate": 0.05},
    como_name="adjacency",
    como_overrides={
        "adjacency_kind": "groups",
        "group_assignments": [f"Group {i // 10 + 1}" for i in range(100)],
        "group_seed": 252,
        "inter_group_weight": 0.2,
        "intra_group_weight": 1.0,
        "mixing_probability": 0.05,
    },
    adjacency_neighbours=0,
)


DEFAULT_SMOOTH_REGRESSION_GROUPS_GOLDILOCKS = Scenario(
    label=(
        "Default (smooth decline) -- regression learning -- group clusters -- "
        "Goldilocks COMO"
    ),
    seed=DEFAULT_SIMPLE_DELAY_REGRESSION_GROUPS_POP_MEAN_GAP.seed + 10,
    horizon=DEFAULT_SIMPLE_DELAY_REGRESSION_GROUPS_POP_MEAN_GAP.horizon,
    n_agents=DEFAULT_SIMPLE_DELAY_REGRESSION_GROUPS_POP_MEAN_GAP.n_agents,
    decision_name="default",
    decision_overrides={
        "rho": 0.4,
        "smu_wb_coeff": -0.5,
        "como_wb_coeff": -0.3,
        "delayed_decline": "smooth",
        "reward_scale": 1.2,
        "penalty_scale": 0.7,
        "fast_timescale": 15.0,
        "slow_timescale": 2.5,
        "effect_horizon": 30,
        "target_total_effect": 0.0,
        "smu_bounds": SMU_CLIP,
        "wellbeing_bounds": WELLBEING_CLIP,
    },
    learning_name="regression_based",
    learning_overrides=dict(
        DEFAULT_SIMPLE_DELAY_REGRESSION_GROUPS_POP_MEAN_GAP.learning_overrides
    ),
    como_name="adjacency_goldilocks",
    como_overrides={
        **DEFAULT_SIMPLE_DELAY_REGRESSION_GROUPS_POP_MEAN_GAP.como_overrides,
        "group_seed": DEFAULT_SIMPLE_DELAY_REGRESSION_GROUPS_POP_MEAN_GAP.seed + 10,
        "width": 1.5,
    },
    adjacency_neighbours=0,
)


SAVED_SCENARIOS: dict[str, Scenario] = {
    NEG_POP_POS_INDCOR.label: NEG_POP_POS_INDCOR,
    NEG_POP_COMO_DISABLED.label: NEG_POP_COMO_DISABLED,
    NEG_POP_POS_INDCOR_POS_COMO.label: NEG_POP_POS_INDCOR_POS_COMO,
    NEG_POP_POS_INDCOR_POS_COMO_SMOOTH.label: NEG_POP_POS_INDCOR_POS_COMO_SMOOTH,
    NEG_POP_POS_INDCOR_RING_MULTI_HOP_SMOOTH.label: NEG_POP_POS_INDCOR_RING_MULTI_HOP_SMOOTH,
    AR1_POP_MEAN_GAP_GRAD_ASCENT.label: AR1_POP_MEAN_GAP_GRAD_ASCENT,
    AR1_POP_POS_GAP_GRAD_ASCENT.label: AR1_POP_POS_GAP_GRAD_ASCENT,
    DEFAULT_SIMPLE_DELAY_REGRESSION_NO_ADJ_POP_POS_GAP.label: DEFAULT_SIMPLE_DELAY_REGRESSION_NO_ADJ_POP_POS_GAP,
    DEFAULT_NO_DELAY_REGRESSION_NO_ADJ_GOLDILOCKS_1_4.label: DEFAULT_NO_DELAY_REGRESSION_NO_ADJ_GOLDILOCKS_1_4,
    DEFAULT_SIMPLE_DELAY_REGRESSION_NO_ADJ_GOLDILOCKS_1_4.label: DEFAULT_SIMPLE_DELAY_REGRESSION_NO_ADJ_GOLDILOCKS_1_4,
    DEFAULT_SIMPLE_DELAY_REGRESSION_RING_POP_MEAN_GAP.label: DEFAULT_SIMPLE_DELAY_REGRESSION_RING_POP_MEAN_GAP,
    DEFAULT_NO_DELAY_REGRESSION_RING_POP_POS_GAP.label: DEFAULT_NO_DELAY_REGRESSION_RING_POP_POS_GAP,
    DEFAULT_NO_DELAY_PEER_RING_POP_POS_GAP.label: DEFAULT_NO_DELAY_PEER_RING_POP_POS_GAP,
    DEFAULT_SIMPLE_DELAY_REGRESSION_RING_GOLDILOCKS_1_4.label: DEFAULT_SIMPLE_DELAY_REGRESSION_RING_GOLDILOCKS_1_4,
    DEFAULT_SIMPLE_DELAY_REGRESSION_GROUPS_POP_MEAN_GAP.label: DEFAULT_SIMPLE_DELAY_REGRESSION_GROUPS_POP_MEAN_GAP,
    DEFAULT_SMOOTH_REGRESSION_GROUPS_GOLDILOCKS.label: DEFAULT_SMOOTH_REGRESSION_GROUPS_GOLDILOCKS,
}
"""Lookup of scenario label to configuration for reusable setups."""


__all__ = [
    "WELLBEING_CLIP",
    "SMU_CLIP",
    "INITIAL_AGENT_PRIORS",
    "Scenario",
    "NEG_POP_POS_INDCOR",
    "NEG_POP_COMO_DISABLED",
    "NEG_POP_POS_INDCOR_POS_COMO",
    "NEG_POP_POS_INDCOR_POS_COMO_SMOOTH",
    "NEG_POP_POS_INDCOR_RING_MULTI_HOP_SMOOTH",
    "AR1_POP_MEAN_GAP_GRAD_ASCENT",
    "AR1_POP_POS_GAP_GRAD_ASCENT",
    "DEFAULT_SIMPLE_DELAY_REGRESSION_NO_ADJ_POP_POS_GAP",
    "DEFAULT_NO_DELAY_REGRESSION_NO_ADJ_GOLDILOCKS_1_4",
    "DEFAULT_SIMPLE_DELAY_REGRESSION_NO_ADJ_GOLDILOCKS_1_4",
    "DEFAULT_SIMPLE_DELAY_REGRESSION_RING_POP_MEAN_GAP",
    "DEFAULT_NO_DELAY_REGRESSION_RING_POP_POS_GAP",
    "DEFAULT_NO_DELAY_PEER_RING_POP_POS_GAP",
    "DEFAULT_SIMPLE_DELAY_REGRESSION_RING_GOLDILOCKS_1_4",
    "DEFAULT_SIMPLE_DELAY_REGRESSION_GROUPS_POP_MEAN_GAP",
    "DEFAULT_SMOOTH_REGRESSION_GROUPS_GOLDILOCKS",
    "SAVED_SCENARIOS",
]
