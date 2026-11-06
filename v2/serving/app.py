"""npl-el v2 예측 API.

실행:  uvicorn app:app --host 0.0.0.0 --port 8000   (v2/serving 폴더에서)
문서:  http://localhost:8000/docs
artifact 위치: 환경변수 NPL_ARTIFACTS_DIR (기본값: 이 파일 옆 artifacts/)
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from predictor import Predictor

ARTIFACTS_DIR = Path(os.environ.get("NPL_ARTIFACTS_DIR", Path(__file__).resolve().parent / "artifacts"))
STATE = {}


@asynccontextmanager
async def lifespan(app):
    STATE["predictor"] = Predictor(ARTIFACTS_DIR)
    yield
    STATE.clear()


app = FastAPI(title="npl-el v2 API", version="0.1.0", lifespan=lifespan,
              description="서울 빌라 NPL 담보의 시세(LightGBM G2)와 회수액·LGD(6줄 계산)를 계산한다.")


# ---------------------------------------------------------------- 입력 스키마 ----
class PropertyIn(BaseModel):
    lat: float = Field(..., ge=37.40, le=37.72, description="위도(서울 범위)")
    lng: float = Field(..., ge=126.75, le=127.20, description="경도(서울 범위)")
    area_m2: float = Field(..., gt=5, le=300, description="전용면적(㎡)")
    build_year: int = Field(..., ge=1950, le=2030, description="건축년도")
    floor: int = Field(..., ge=-5, le=60, description="층(지하는 음수)")
    gu: str = Field(..., min_length=2, description="구 이름. 예: 마포구")
    dong: str = Field(..., min_length=1, description="법정동 이름. 예: 망원동")
    jibun: str = Field(..., min_length=1, description="지번(국토부 실거래가 형식). 예: 123-4")
    as_of: str = Field(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="기준 연월 YYYY-MM")

    @model_validator(mode="after")
    def build_year_not_future(self):
        if self.build_year > int(self.as_of[:4]):
            raise ValueError("build_year가 기준 연도보다 늦다")
        return self

    def to_item(self):
        d = self.model_dump()
        y, m = d.pop("as_of").split("-")
        d["as_of_year"], d["as_of_month"] = int(y), int(m)
        return d


class RecoveryIn(PropertyIn):
    deposit_won: float = Field(0, ge=0, description="선순위 임차보증금(원)")
    ead_won: Optional[float] = Field(None, gt=0, description="대출원금 EAD(원). 비우면 시세×0.70 가정")
    months: list[int] = Field([12, 24, 36], min_length=1, max_length=6, description="매각까지 개월 수")

    @field_validator("months")
    @classmethod
    def months_range(cls, v):
        if any(m < 1 or m > 120 for m in v):
            raise ValueError("months는 1~120 사이여야 한다")
        return v


# ---------------------------------------------------------------- 엔드포인트 ----
def _predictor():
    p = STATE.get("predictor")
    if p is None:
        raise HTTPException(503, "모델이 로드되지 않았다")
    return p


@app.get("/health")
def health():
    p = _predictor()
    m = p.meta
    return {
        "status": "ok",
        "model": m["model_name"],
        "model_version": m["model_version"],
        "train_period": [m["train_start"], m["train_end"]],
        "test_mdape_pct": m["test_mdape_pct"],
        "n_reference": len(p.ref_ppm2),
    }


@app.post("/v1/price")
def price(req: PropertyIn):
    out = _predictor().predict_price(req.to_item())
    out["price_manwon"] = round(out["price_manwon"], 1)
    out["price_won"] = round(out["price_won"])
    return out


@app.post("/v1/recovery")
def recovery(req: RecoveryIn):
    out = _predictor().predict_recovery(req.to_item(), req.deposit_won, req.ead_won, req.months)
    base = [r for r in out["grid"] if r["rho_scenario"] == "기본"]
    return {
        "price_won": round(out["price"]["price_won"]),
        "ead_won": round(out["ead_won"]),
        "ead_assumed": out["ead_assumed"],
        "summary_base_rho": [{"months": r["months"], "LGD": r["LGD"], "PV": round(r["PV"])} for r in base],
        "grid": out["grid"],
        "neighbors": out["price"]["neighbors"],
        "flags": out["flags"],
    }
