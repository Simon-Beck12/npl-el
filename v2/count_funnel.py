"""v2 가격 모델 데이터 단계별 건수 집계. 기존 .py는 수정하지 않고 import만 한다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "v2"))

from features import load_merged, build_features  # noqa: E402
from split import add_contract_date, TEST_END  # noqa: E402
from price_model import v2_filter, load_v2_split  # noqa: E402

RESULTS_MD = Path(__file__).resolve().parent / "results" / "funnel_counts.md"


def main():
    merged = load_merged()
    featured, n_missing = build_features(merged)
    total = len(featured)

    after_direct = int((featured["direct_flag"] == 0).sum())
    after_outlier = int((featured["outlier_flag"] == 0).sum())
    both_excluded = int(((featured["direct_flag"] == 1) & (featured["outlier_flag"] == 1)).sum())

    filtered = v2_filter(featured)
    dated = add_contract_date(filtered)
    unused_future = int((dated["contract_date"] > TEST_END).sum())

    train_all, core, val, test = load_v2_split()

    rows = [
        ("정제(SYN-001 전체)", total, "필터 없음, rh_trade_clean.parquet 전체"),
        ("직거래 제외 후(direct_flag=0)", after_direct, "정제 중 direct_flag=0만(상하위 1% 필터는 아직 적용 안 함)"),
        ("상하위 1% 제외 후(outlier_flag=0)", after_outlier, "정제 중 outlier_flag=0만(직거래 필터는 아직 적용 안 함)"),
        ("직거래·상하위1% 중복 제외 건수", both_excluded, "direct_flag=1 이고 outlier_flag=1 인 행(두 필터 모두에 걸림)"),
        ("학습(core)", len(core), "2024-11-01~2026-02월말(VAL_START 이전), v2 필터(직거래·상하위1% 제외) 적용 후"),
        ("검증(val)", len(val), "2026-03-01~2026-05-31, v2 필터 적용 후"),
        ("학습+검증(train_all)", len(train_all), "2024-11-01~2026-05-31(=core+val), v2 필터 적용 후"),
        ("테스트(test)", len(test), "2026-06-01~2026-08-31, v2 필터 적용 후"),
        ("2026-09~10(미사용)", unused_future, "2026-09-01 이후 계약, v2 필터 적용 후(학습·검증·테스트 어디에도 안 씀)"),
    ]

    print(f"{'항목':30s} {'건수':>10s}  설명")
    for name, n, note in rows:
        print(f"{name:30s} {n:>10,d}  {note}")

    assert len(test) == 7561, f"테스트 건수가 7,561이 아님: {len(test)} — 멈춤"
    print(f"\n테스트 건수 확인: {len(test)} == 7,561 통과")

    lines = [
        "# v2 가격 모델 데이터 단계별 건수(funnel)",
        "",
        "`v2/price_model.py`의 `load_v2_split()`, `v2_filter()`를 그대로 호출해 센 건수다. "
        "모델은 다시 학습하지 않았고, 기존 결과 파일의 숫자도 바꾸지 않았다.",
        "",
        "| 항목 | 건수 | 기간·조건 |",
        "|---|---|---|",
    ]
    for name, n, note in rows:
        lines.append(f"| {name} | {n:,} | {note} |")
    lines += [
        "",
        f"지표 종류: 행 건수. 임계값: 테스트(test) 건수가 7,561이 아니면 멈춤 — **통과(7,561)**. "
        "집계 방식: `data/processed/rh_trade_clean.parquet`(69,371행)에 `src/features.py`의 "
        "`load_merged`+`build_features`를 적용한 뒤(좌표 결측 0건), `v2/price_model.py`의 "
        "`v2_filter`+`load_v2_split` 그대로 호출.",
        "",
        "결과 파일: `v2/count_funnel.py` 콘솔 출력.",
    ]
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {RESULTS_MD}")


if __name__ == "__main__":
    main()
