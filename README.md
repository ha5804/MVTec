# PatchCore Heatmap Experiments

PatchCore가 MVTec AD와 VisA의 어떤 category/defect type에서 약한지 heatmap으로 보기 위한 구조입니다.

## File Structure

```text
config/config.py          # 실험에서 바꿀 값은 여기만 수정
experiments.py            # Jupyter에서 import하는 공통 실행 함수
datasets/anomaly.py       # MVTec/VisA 공통 dataset wrapper
models/patchcore.py       # PatchCore
utils/visualization.py    # image/heatmap/overlay 출력
utils/metrics.py          # image-level AUC
notebooks/patchcore_heatmap.ipynb
```

## Jupyter Workflow

1. `config/config.py`에서 `DATASET`, `CATEGORY`, `TRAIN_LIMIT`, `TEST_LIMIT_PER_CLASS` 등을 바꿉니다.
2. 노트북에서 config를 reload합니다.
3. `run_experiment(cfg)`를 실행하고 `show_result(test_data, model, index=...)`로 heatmap을 확인합니다.

```python
import importlib
import config.config as cfg
from experiments import run_experiment
from utils.visualization import show_result

importlib.reload(cfg)

result = run_experiment(cfg, show_index=cfg.TEST_INDEX, compute_auc=True)
model = result["model"]
test_data = result["test_data"]

show_result(test_data, model, index=0)
```

## Config Examples

MVTec:

```python
DATASET = "mvtec"
CATEGORY = "bottle"
```

VisA:

```python
DATASET = "visa"
CATEGORY = "candle"
```

Small limits are intentionally supported because heatmap inspection is usually more useful after quick category-by-category runs:

```python
TRAIN_LIMIT = 50
TEST_LIMIT = None
TEST_LIMIT_PER_CLASS = 3
```

