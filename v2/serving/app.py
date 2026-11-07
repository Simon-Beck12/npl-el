"""EXP-011 7단계: v2 시세·회수액 서빙 FastAPI 앱.

v2/recovery.py(compute_recovery_v2)를 그대로 import한다. 수정하지 않는다.
지오코딩은 이번 범위에 없다 — lat/lng은 호출자가 직접 준다.
"""
import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "v2"))

from recovery import compute_recovery_v2  # noqa: E402

from model_loader import load_model, KnnIndex  # noqa: E402
from valuation_core import build_target_row, predict_valuation_batch  # noqa: E402

app = FastAPI(title="npl-el v2 serving", version="1")

# EXP-010(results/exp010_area_filter.md 표, 일괄매각 의심 제외 후 rho) 값을 그대로 쓴다.
RHO_SCENARIOS = {"비관": 0.5738, "기본": 0.644620, "낙관": 0.8002}

_booster, _ref, _meta = load_model()
# EXP-014 9단계: cKDTree·건물별 개수를 ref로 1번만 만든다(요청마다 새로 만들지 않는다).
_knn_index = KnnIndex(_ref)


class ValuationRequest(BaseModel):
    lat: float = Field(ge=37.4, le=37.8)
    lng: float = Field(ge=126.7, le=127.3)
    excluUseAr: float = Field(gt=0)
    floor: int
    buildYear: int
    dealYear: int
    dealMonth: int = Field(ge=1, le=12)
    # EXP-013: same_bldg 계산에 주소가 꼭 필요해 필수 입력으로 바꾼다. 빈 문자열도 막는다.
    gu_name: str = Field(min_length=1)
    umdNm: str = Field(min_length=1)
    jibun: str = Field(min_length=1)


class ValuationResponse(BaseModel):
    market_price: float
    price_per_m2: float
    knn_ppm2_5: float
    same_bldg: int
    model_version: str


class RecoveryRequest(BaseModel):
    market_price: float = Field(gt=0)
    rho: Optional[float] = Field(default=None, gt=0)
    rho_scenario: Optional[Literal["비관", "기본", "낙관"]] = None
    deposit: float = Field(ge=0)
    ead: float = Field(gt=0)
    months: int = Field(ge=1, le=120)

    @model_validator(mode="after")
    def check_rho(self):
        if self.rho is None and self.rho_scenario is None:
            raise ValueError("rho 또는 rho_scenario 중 하나는 반드시 줘야 한다")
        if self.rho is not None and self.rho_scenario is not None:
            raise ValueError("rho와 rho_scenario를 동시에 줄 수 없다")
        return self


class RecoveryResponse(BaseModel):
    S: float
    K: float
    P: float
    D2: float
    R: float
    PV: float
    EAD: float
    LGD: Optional[float]
    rho_used: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": app.version,
        "model_name": _meta.get("registered_model_name"),
        "model_registry_version": _meta.get("version"),
        "model_alias": _meta.get("alias"),
        "run_id": _meta.get("run_id"),
        "test_mdape": _meta.get("test_mdape"),
        "meta": _meta,
    }


@app.post("/valuation", response_model=ValuationResponse)
def valuation(req: ValuationRequest):
    target = build_target_row(
        lat=req.lat, lng=req.lng, excluUseAr=req.excluUseAr, floor=req.floor,
        buildYear=req.buildYear, dealYear=req.dealYear, dealMonth=req.dealMonth,
        gu_name=req.gu_name, umdNm=req.umdNm, jibun=req.jibun,
    )
    result_df = predict_valuation_batch(target, _knn_index, _booster)
    result = {
        "market_price": float(result_df["market_price"].iloc[0]),
        "price_per_m2": float(result_df["price_per_m2"].iloc[0]),
        "knn_ppm2_5": float(result_df["knn_ppm2_5"].iloc[0]),
        "same_bldg": int(result_df["same_bldg"].iloc[0]),
    }
    return ValuationResponse(**result, model_version=app.version)


@app.post("/recovery", response_model=RecoveryResponse)
def recovery(req: RecoveryRequest):
    rho_used = req.rho if req.rho is not None else RHO_SCENARIOS[req.rho_scenario]
    try:
        result = compute_recovery_v2(
            market_price=req.market_price, rho=rho_used,
            deposit=req.deposit, ead=req.ead, months=req.months,
        )
    except Exception as exc:  # pragma: no cover - 6줄 계산은 산술 연산뿐이라 평소엔 발생하지 않음
        raise HTTPException(status_code=422, detail=str(exc))
    return RecoveryResponse(**result, rho_used=rho_used)
