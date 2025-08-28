# Fantasy Football Codebase Refactoring Plan

## 🎯 Goals
- **Clean separation of concerns** - Each module has a single responsibility
- **Easy to navigate** - Clear directory structure and naming
- **Simple imports** - Straightforward dependency management
- **Clear entry points** - Examples for every use case
- **Maintainable** - Easy to extend and modify

## 🏗️ New Project Structure

```
fantasy-football/
├── src/                          # Source code
│   ├── __init__.py
│   ├── core/                     # Core domain models
│   │   ├── __init__.py
│   │   ├── player.py             # Player, PlayerPool classes
│   │   ├── draft.py              # DraftState, League, DraftSettings
│   │   └── scoring.py            # Scoring systems (PPR, standard, etc.)
│   ├── models/                   # Prediction models
│   │   ├── __init__.py
│   │   ├── rookie.py             # Rookie prediction models
│   │   ├── injury.py             # Injury risk modeling
│   │   └── valuation.py          # VORP calculations, projections
│   ├── strategies/               # Draft strategies
│   │   ├── __init__.py
│   │   ├── mcts.py               # MCTS implementation
│   │   ├── injury_aware.py       # Injury-aware strategies
│   │   └── traditional.py        # ADP, VORP strategies
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── data_loader.py        # Data loading and preprocessing
│       ├── visualization.py      # Plotting and charts
│       └── preprocessing.py      # Data cleaning utilities
├── data/                         # Data files
│   ├── raw/                      # Original data files
│   │   ├── draft_board.csv
│   │   ├── adp_rankings.csv
│   │   └── rookie_data.csv
│   ├── processed/                # Cleaned data files
│   │   ├── player_pool.csv
│   │   └── injury_enhanced.csv
│   └── models/                   # Trained model files
│       └── rookie_regressor.pkl
├── examples/                     # Example scripts
│   ├── basic_draft.py            # Simple MCTS draft example
│   ├── injury_aware_draft.py     # Injury-enhanced draft
│   ├── rookie_prediction.py      # Rookie modeling demo
│   └── strategy_comparison.py    # Compare different strategies
├── docs/                         # Documentation
│   ├── README.md                 # Main documentation
│   ├── API.md                    # API reference
│   └── tutorials/                # How-to guides
├── tests/                        # Unit tests
│   ├── test_core.py
│   ├── test_models.py
│   └── test_strategies.py
├── deployment/                   # Deployment configurations
│   ├── colab/                    # Google Colab packages
│   │   ├── basic_mcts.zip
│   │   └── injury_enhanced.zip
│   └── jupyter/                  # Jupyter notebooks
│       └── interactive_draft.ipynb
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
└── README.md                     # Project overview
```

## 🔄 Key Changes

### 1. **Clean Module Organization**
- **src/core/** - Domain models (Player, Draft, League)
- **src/models/** - ML models (Rookie, Injury, Valuation)  
- **src/strategies/** - Draft strategies (MCTS, Traditional)
- **src/utils/** - Shared utilities

### 2. **Data Organization**
- **data/raw/** - Original data files
- **data/processed/** - Cleaned, ready-to-use data
- **data/models/** - Trained model artifacts

### 3. **Clear Examples**
- **examples/basic_draft.py** - Simple getting started
- **examples/injury_aware_draft.py** - Advanced injury features
- **examples/rookie_prediction.py** - ML model demo
- **examples/strategy_comparison.py** - Compare approaches

### 4. **Professional Structure**
- **tests/** - Unit tests for all modules
- **docs/** - Comprehensive documentation
- **deployment/** - Ready-to-use packages (Colab, Jupyter)

## 📦 Core Module Design

### src/core/player.py
```python
@dataclass
class Player:
    name: str
    position: str
    team: str
    vorp: float
    adp_rank: float
    projections: Dict[str, float]
    injury_data: Optional[Dict] = None

class PlayerPool:
    def __init__(self, players: List[Player])
    def filter_by_position(self, position: str) -> List[Player]
    def get_top_players(self, n: int) -> List[Player]
    def add_injury_data(self, injury_data: Dict)
```

### src/strategies/mcts.py
```python
class MCTSStrategy:
    def __init__(self, reward_function, opponent_model, simulations=400)
    def search(self, draft_state: DraftState) -> Player
    def make_pick(self, draft_state: DraftState) -> Player

class InjuryAwareMCTS(MCTSStrategy):
    def __init__(self, injury_weight: float = 0.3, **kwargs)
```

### src/models/rookie.py
```python
class RookiePredictor:
    def __init__(self, model_type: str = 'random_forest')
    def train(self, data: pd.DataFrame) -> None
    def predict(self, rookie_data: Dict) -> float
    def get_feature_importance(self) -> Dict[str, float]
```

## 🚀 Migration Benefits

### Before (Current Issues)
- ❌ Files scattered across root directory
- ❌ Complex import paths
- ❌ Duplicate code in multiple files
- ❌ Hard to find relevant functionality
- ❌ Jupyter notebook with 2600+ lines

### After (Refactored)
- ✅ **Clean module separation** - Easy to find what you need
- ✅ **Simple imports** - `from src.strategies import MCTSStrategy`
- ✅ **DRY code** - Shared utilities, no duplication
- ✅ **Clear entry points** - Examples for every use case
- ✅ **Maintainable** - Easy to extend and test

## 📋 Implementation Steps

1. **Create new directory structure**
2. **Extract core classes** from existing files
3. **Refactor modules** into clean, focused files  
4. **Create simple examples** for each use case
5. **Update imports** and dependencies
6. **Create comprehensive tests**
7. **Update documentation**

## 🎯 User Experience

### Simple Draft (2 lines)
```python
from src.strategies import MCTSStrategy
from src.utils import load_default_data

strategy = MCTSStrategy()
pick = strategy.make_pick(draft_state)
```

### Injury-Aware Draft (3 lines)
```python
from src.strategies import InjuryAwareMCTS
from src.utils import load_injury_enhanced_data

strategy = InjuryAwareMCTS(injury_weight=0.3)
pick = strategy.make_pick(draft_state)
```

### Rookie Prediction (3 lines)
```python
from src.models import RookiePredictor

predictor = RookiePredictor()
projection = predictor.predict(rookie_data)
```

This refactoring will make the codebase **professional, maintainable, and easy to use**! 🏈
