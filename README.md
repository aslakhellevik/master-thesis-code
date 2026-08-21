# master-thesis-code

Code repository for the master's thesis "Network Effects of Social Media Use on Well-Being: From Agent-Based Models to Statistical Inference". Each notebook is self-contained and reproduces all figures and tables for its corresponding thesis chapter.

Author: Aslak Hellevik.

The published thesis (PDF) is openly available at
[NVA](https://nva.sikt.no/registration/019fdb8244e4-e2b85e96-eac5-46e1-a84f-bf7f265abef4),
and the LaTeX source lives in
[master-thesis-latex](https://github.com/aslakhellevik/master-thesis-latex).

## Repository layout

```
master-thesis-code/
├── abm/                          # Agent-based model (Chapter 3)
│   ├── results_for_chapter_3.ipynb
│   ├── core_abm_class.py
│   ├── como_functions.py
│   ├── decision_rules.py
│   ├── learning_rules.py
│   └── scenario_library.py
├── results_for_chapter_4.ipynb   # Recursive model (Chapter 4)
├── results_for_chapter_6.ipynb   # Monte Carlo validation (Chapter 6)
├── results_for_chapter_7.ipynb   # Simultaneous estimation (Chapter 7)
├── figures_for_thesis/           # All generated figures (23 PDFs)
├── requirements.txt
├── LICENSE
└── README.md
```

### Results notebooks

| Notebook | Chapter | Self-contained | Figures | Tables |
|---|---|---|---|---|
| `abm/results_for_chapter_3.ipynb` | 3: ABM Investigation | No (imports ABM modules in `abm/`) | 8 | 2 |
| `results_for_chapter_4.ipynb` | 4: Two-Step Model | Yes | 2 | 0 |
| `results_for_chapter_6.ipynb` | 6: Monte Carlo Validation | Yes | 9 | 12 |
| `results_for_chapter_7.ipynb` | 7: Simultaneous Systems | Yes | 3 | 4 |

Chapters 4, 6, and 7 notebooks use only standard scientific Python packages and can be run independently. Chapter 4 requires only NumPy and Matplotlib. Chapters 6 and 7 additionally use SciPy, pandas, and joblib for parallelised Monte Carlo simulations. Chapter 3 imports the ABM framework modules located alongside it in `abm/` and also requires NetworkX for graph construction.

### ABM framework (`abm/`)

The agent-based model used in Chapter 3 consists of five modules:

- **`core_abm_class.py`** -- Agent data classes (`AgentConfig`, `AgentState`, `AgentStepContext`, `AgentStepResult`, `SimulationLog`), the `BaseAgent` class, the `ABDRunner` simulation driver, and the `assign_learning_rules_by_fraction()` utility.
- **`como_functions.py`** -- Cost of Missing Out (COMO) specifications in three families (population-mean, adjacency-weighted, and Goldilocks/quadratic), each with unbounded, non-negative, and softplus variants. Also provides adjacency matrix builders (`build_simple_adjacency`, `build_ring_adjacency`, `build_grouped_adjacency`, `expand_adjacency_with_hops`).
- **`decision_rules.py`** -- Decision rule factories (`AR1_decision`, `default_decision`, `smooth_smu_with_spillover_decision`) that determine how agents update SMU and wellbeing each period, with support for delayed-effect kernels and neighbour spillover.
- **`learning_rules.py`** -- Learning rule factories (`regression_based_learning`, `peer_learning`, `gradient_ascent_learning`) that govern how agents adapt their behaviour.
- **`scenario_library.py`** -- Shared constants (`WELLBEING_CLIP`, `SMU_CLIP`, `INITIAL_AGENT_PRIORS`), the `Scenario` dataclass, and 15 pre-configured scenario presets stored in the `SAVED_SCENARIOS` dictionary.

## Reproducing results

### Dependencies

```bash
pip install -r requirements.txt
```

Tested with Python 3.13.5 (Anaconda).

### Running the notebooks

Each notebook saves its figures to `figures_for_thesis/`. Run them from the repository root:

```bash
# Chapter 3 (must run from abm/ directory for module imports)
cd abm && jupyter notebook results_for_chapter_3.ipynb

# Chapters 4, 6, 7 (run from repo root)
jupyter notebook results_for_chapter_6.ipynb
```

Chapters 6 and 7 run parallelised Monte Carlo simulations and may take several hours depending on the number of CPU cores available.

## License

Released under the MIT License — see `LICENSE`.
