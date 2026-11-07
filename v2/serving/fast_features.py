"""EXP-014 6단계: 서빙 경로 전용 k-NN·같은 건물 인덱스(KnnIndex).

문제: v2/price_model.py의 add_knn_ppm2·add_same_bldg는 호출할 때마다 cKDTree와 건물별
개수를 기준 데이터(ref) 전체로 새로 만든다. 서빙에서는 ref가 모델 로드 시점에 고정되는데도
요청·행마다 이 재계산이 반복된다(n_ref=45,634행 기준 병목). 이 모듈은 ref로 1번만 인덱스를
만들어 두고 여러 행을 한 번에 조회한다. price_model.py는 수정하지 않는다.

세부 내용 — add_same_bldg 규칙 확인(v2/price_model.py:116-125):
- bldg_key(df) = gu_name(astype str) + "|" + umdNm(astype str, strip) + "|" + jibun(astype str, strip).
  주소 정규화는 이 함수(price_model.bldg_key)를 그대로 import해 재사용한다(재구현하지 않음 → 규칙이
  어긋날 여지가 없다).
- add_same_bldg(target, ref, exclude_self)는 ref의 bldg_key별 등장 횟수(value_counts)를 세고,
  threshold = exclude_self면 2, 아니면 1로 두고 target 키의 개수가 threshold 이상이면 1이다.
- 서빙(valuation_core.predict_valuation)은 항상 exclude_self=False로 호출한다(ref=학습 데이터,
  target=새 요청이라 자기 자신을 뺄 필요가 없다) → threshold=1: target 주소와 같은 주소의 ref
  거래가 1건이라도 있으면 same_bldg=1. 이 클래스는 이 경로(exclude_self=False)만 구현한다.

세부 내용 — add_knn_ppm2 규칙 확인(v2/price_model.py:128-149, exclude_self=False):
ref 전체 좌표(loc_x_km, loc_y_km)로 cKDTree를 만들고, k=5 최근접 이웃의 price_per_m2 중앙값을
쓴다(자기 제외 로직 없음). cKDTree는 여러 점을 한 번에 질의할 수 있어 ref로 1번만 만들면 된다.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "v2"))

from price_model import bldg_key, KNN_K  # noqa: E402


class KnnIndex:
    """ref(기준 거래 데이터)로 1번만 만들어 재사용한다. exclude_self=False 경로만 지원한다
    (서빙에서 쓰는 유일한 경로 — v2/price_model.py의 build_lgbm_frame은 학습 때만 exclude_self=True를 쓴다)."""

    def __init__(self, ref, k=KNN_K):
        ref = ref.reset_index(drop=True)
        self.k = k
        self.tree = cKDTree(ref[["loc_x_km", "loc_y_km"]].to_numpy())
        self.ref_price = ref["price_per_m2"].to_numpy()
        self.bldg_counts = bldg_key(ref).value_counts()

    def knn_ppm2(self, target):
        """target: loc_x_km·loc_y_km 컬럼이 있는 DataFrame(1행 이상).
        add_knn_ppm2(target, ref, exclude_self=False)와 같은 값을 반환한다."""
        _, idx = self.tree.query(target[["loc_x_km", "loc_y_km"]].to_numpy(), k=self.k)
        if self.k == 1:
            idx = idx[:, None]
        return np.median(self.ref_price[idx], axis=1)

    def same_bldg(self, target):
        """target: gu_name·umdNm·jibun 컬럼이 있는 DataFrame.
        add_same_bldg(target, ref, exclude_self=False)와 같은 값을 반환한다."""
        key_t = bldg_key(target)
        return (key_t.map(self.bldg_counts).fillna(0) >= 1).astype(int).to_numpy()
