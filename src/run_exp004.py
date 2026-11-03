"""EXP-004: FAISS 탐색 범위 확장 (작은 k, 큰 위치 가중치).

knn_comps.py는 수정하지 않는다. knn_comps 모듈의 탐색 범위 전역 변수(K_GRID, W_LOC_GRID,
W_OTHER_GRID)와 이웃 저장 개수(NEIGHBOR_STORE_N)를 이 스크립트에서 덮어써, knn_comps.py의
tune_fold/run_fold 함수를 그대로 재사용한다.
"""

from pathlib import Path

import pandas as pd

import knn_comps
from features import load_merged, build_features
from split import split_dv02

ROOT = Path(__file__).resolve().parent.parent
PRED_PARQUET = ROOT / "data" / "processed" / "exp004_predictions.parquet"
TUNING_CSV = ROOT / "data" / "processed" / "exp004_fold_tuning.csv"

# EXP-004 탐색 범위 (실행 전 고정, [실험 등록 내용] 그대로)
K_GRID = [1, 2, 3, 5]
W_LOC_GRID = [4.0, 8.0, 16.0]
W_OTHER_GRID = [0.5, 1.0, 2.0]


def main():
    knn_comps.K_GRID = K_GRID
    knn_comps.W_LOC_GRID = W_LOC_GRID
    knn_comps.W_OTHER_GRID = W_OTHER_GRID

    n_combos = len(list(knn_comps.grid_combos()))
    print(f"탐색 조합 수: {n_combos:,} (기대값 324)")
    assert n_combos == 324

    merged = load_merged()
    featured, n_missing = build_features(merged)
    assert n_missing / len(featured) <= 0.01, "좌표 없는 행이 1%를 넘음 — 멈춤 지점"

    folds = split_dv02(featured)

    tuning_rows = []
    pred_rows = []
    total_future_violations = 0

    for name, train, test in folds:
        best, core_n, val_n = knn_comps.tune_fold(train)
        print(f"{name} 튜닝: core={core_n:,}, val={val_n:,}, 최적={best}")

        # 이웃 10건 대신 그 fold에서 선택된 k건만 저장한다 (knn_comps.py 미수정, 전역만 덮어씀)
        knn_comps.NEIGHBOR_STORE_N = int(best["k"])
        test_pred, future_violations = knn_comps.run_fold(name, train, test, best)
        test_mdape, _ = knn_comps.mdape(test_pred["actual_price"].to_numpy(), test_pred["faiss_pred"].to_numpy())
        print(f"{name} 테스트 MdAPE={test_mdape:.2f} (n={len(test_pred):,}), 미래 이웃 위반={future_violations}")

        total_future_violations += future_violations
        tuning_rows.append({
            "fold": name, **best, "test_mdape": test_mdape, "core_n": core_n, "val_n": val_n, "test_n": len(test_pred),
        })
        pred_rows.append(test_pred)

    assert total_future_violations == 0, f"미래 이웃 {total_future_violations}건 발견 — 멈춤 지점"

    predictions = pd.concat(pred_rows, ignore_index=True)
    PRED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PRED_PARQUET, index=False)
    print(f"저장: {PRED_PARQUET} ({len(predictions):,}행)")

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(TUNING_CSV, index=False)
    print(f"저장: {TUNING_CSV}")
    print(tuning_df.to_string(index=False))

    pooled_mdape, _ = knn_comps.mdape(predictions["actual_price"].to_numpy(), predictions["faiss_pred"].to_numpy())
    print(f"\npooled EXP-004 FAISS MdAPE: {pooled_mdape:.2f} (n={len(predictions):,})")

    return predictions, tuning_df


if __name__ == "__main__":
    main()
