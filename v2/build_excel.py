"""EXP-009 단계 9: v2/results/npl_v2_review.xlsx (3시트: 가정/검토표/요약).

모든 금액은 만원 단위. 검토표의 계산 열은 전부 가정 시트를 참조하는 수식이다.
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.worksheet.datavalidation import DataValidation

V2_ROOT = Path(__file__).resolve().parent
PORTFOLIO_V2_PARQUET = V2_ROOT / "data" / "processed" / "portfolio_v2.parquet"
OUT_XLSX = V2_ROOT / "results" / "npl_v2_review.xlsx"

RHO = {"비관": 0.5740926374276487, "기본": 0.644620490703548, "낙관": 0.8042342367622282}
MANWON_FMT = "#,##0.0"
PCT_FMT = "0.0%"
INPUT_FONT = Font(color="0000FF")
FORMULA_FONT = Font(color="000000")


def set_cell(ws, addr, value, kind="formula", number_format=None, formula=None):
    cell = ws[addr]
    cell.value = formula if formula is not None else value
    cell.font = INPUT_FONT if kind == "input" else FORMULA_FONT
    if number_format:
        cell.number_format = number_format
    return cell


def add_list_validation(ws, addr, options_csv):
    dv = DataValidation(type="list", formula1=f'"{options_csv}"', allow_blank=False)
    dv.sqref = addr
    ws.add_data_validation(dv)


def build_assumptions(wb):
    ws = wb.create_sheet("가정")
    set_cell(ws, "A1", "가정값 (이 시트에서 바꿀 수 있다) — EXP-009, 모두 만원 단위", None)
    rows = [
        ("LTV", 0.70, "v2 시세 기준 대출원금 비율"),
        ("집행비용율", 0.01, "매각가의 1%"),
        ("최우선변제_보증금한도_만원", 16500, "서울 소액임차인 최우선변제 현행 기준표"),
        ("최우선변제_최대지급_만원", 5500, "서울 소액임차인 최우선변제 현행 기준표"),
        ("최우선변제_매각가비율한도", 0.5, "최우선변제 합계는 매각가의 1/2 한도"),
        ("할인율", 0.08, "연 8%"),
        ("깡통_보증금비율", 0.9, "보증금_깡통 = v2시세 × 이 비율"),
    ]
    for i, (name, value, note) in enumerate(rows, start=2):
        set_cell(ws, f"A{i}", name, None)
        set_cell(ws, f"B{i}", value, "input")
        set_cell(ws, f"C{i}", note, None)

    set_cell(ws, "A10", "낙찰가율 분위수(학습 대상 83건, 포트폴리오 낙찰 12건 제외)", None)
    set_cell(ws, "A11", "비관(25%)", None)
    set_cell(ws, "B11", RHO["비관"], "input")
    set_cell(ws, "A12", "기본(50%)", None)
    set_cell(ws, "B12", RHO["기본"], "input")
    set_cell(ws, "A13", "낙관(75%)", None)
    set_cell(ws, "B13", RHO["낙관"], "input")

    set_cell(ws, "A15", "선택(드롭다운)", None)
    set_cell(ws, "A16", "낙찰가율시나리오", None)
    set_cell(ws, "B16", "기본", "input")
    set_cell(ws, "A17", "보증금시나리오", None)
    set_cell(ws, "B17", "기본", "input")
    set_cell(ws, "A18", "매각시점개월", None)
    set_cell(ws, "B18", 24, "input")
    set_cell(ws, "A19", "선택된 낙찰가율(계산)", None)
    set_cell(ws, "B19", None, formula='=CHOOSE(MATCH(B16,{"비관","기본","낙관"},0),B11,B12,B13)')

    set_cell(ws, "A21", "포트폴리오에서 제외된 2건(v2 시세 추정 실패)", None)
    set_cell(ws, "A22", "2024-0600-045986 동대문구 답십리동 75-** (낙찰)", None)
    set_cell(ws, "A23", "2024-1000-080088 중랑구 망우동 518-** (미낙찰)", None)
    set_cell(ws, "A24", "이유: 최초 개찰일이 2024-11로, 그 이전 거래 데이터가 0건(rh_trade 수집 시작월과 같음)", None)

    for col, w in [("A", 42), ("B", 12), ("C", 50)]:
        ws.column_dimensions[col].width = w
    add_list_validation(ws, "B16", "비관,기본,낙관")
    add_list_validation(ws, "B17", "없음,기본,깡통")
    add_list_validation(ws, "B18", "12,24,36")
    return ws


def build_review(wb, port):
    ws = wb.create_sheet("검토표")
    headers = ["cltrMngNo", "masked_주소", "구", "결과구분", "건물면적_m2", "v2시세_만원",
               "EAD_만원", "보증금_없음_만원", "보증금_기본_만원", "보증금_깡통_만원", "선택보증금_만원",
               "선택낙찰가율", "예상매각가S_만원", "집행비용K_만원", "최우선변제조건", "최우선변제P_만원",
               "보증금잔액D2_만원", "남은금액_만원", "회수액R_만원", "PV_만원", "LGD"]
    for j, h in enumerate(headers, start=1):
        from openpyxl.utils import get_column_letter
        set_cell(ws, f"{get_column_letter(j)}1", h, None)

    for i, (_, row) in enumerate(port.iterrows(), start=2):
        set_cell(ws, f"A{i}", row["cltrMngNo"], "input")
        set_cell(ws, f"B{i}", row["masked_주소"], "input")
        set_cell(ws, f"C{i}", row["구"], "input")
        set_cell(ws, f"D{i}", row["결과구분"], "input")
        set_cell(ws, f"E{i}", round(row["건물면적"], 2), "input")
        set_cell(ws, f"F{i}", round(row["v2_price"] / 10000, 1), "input", MANWON_FMT)
        set_cell(ws, f"G{i}", None, formula=f"=F{i}*가정!$B$2", number_format=MANWON_FMT)
        set_cell(ws, f"H{i}", 0, "input", MANWON_FMT)
        set_cell(ws, f"I{i}", round(row["보증금_기본_v2"] / 10000, 1), "input", MANWON_FMT)
        set_cell(ws, f"J{i}", None, formula=f"=F{i}*가정!$B$8", number_format=MANWON_FMT)
        set_cell(ws, f"K{i}", None,
                  formula=f'=CHOOSE(MATCH(가정!$B$17,{{"없음","기본","깡통"}},0),H{i},I{i},J{i})',
                  number_format=MANWON_FMT)
        set_cell(ws, f"L{i}", None, formula="=가정!$B$19")
        set_cell(ws, f"M{i}", None, formula=f"=F{i}*L{i}", number_format=MANWON_FMT)
        set_cell(ws, f"N{i}", None, formula=f"=M{i}*가정!$B$3", number_format=MANWON_FMT)
        set_cell(ws, f"O{i}", None, formula=f"=K{i}<=가정!$B$4")
        set_cell(ws, f"P{i}", None, formula=f"=IF(O{i},MIN(K{i},가정!$B$5,M{i}*가정!$B$6),0)", number_format=MANWON_FMT)
        set_cell(ws, f"Q{i}", None, formula=f"=K{i}-P{i}", number_format=MANWON_FMT)
        set_cell(ws, f"R{i}", None, formula=f"=M{i}-N{i}-P{i}-Q{i}", number_format=MANWON_FMT)
        set_cell(ws, f"S{i}", None, formula=f"=MAX(0,MIN(G{i},R{i}))", number_format=MANWON_FMT)
        set_cell(ws, f"T{i}", None, formula=f"=S{i}/(1+가정!$B$7)^(가정!$B$18/12)", number_format=MANWON_FMT)
        set_cell(ws, f"U{i}", None, formula=f"=1-T{i}/G{i}", number_format=PCT_FMT)

    for col, w in [("A", 18), ("B", 20)]:
        ws.column_dimensions[col].width = w
    return ws, len(port)


def build_summary(wb, n_rows):
    ws = wb.create_sheet("요약", 0)
    last = 1 + n_rows
    set_cell(ws, "A1", "EXP-009 v2 포트폴리오 회수액·LGD 요약(모두 수식, 가정 시트의 선택을 따른다)", None)
    set_cell(ws, "A3", "18건 합계 EAD(만원)", None)
    set_cell(ws, "B3", None, formula=f"=SUM(검토표!$G$2:$G${last})", number_format=MANWON_FMT)
    set_cell(ws, "A4", "18건 합계 PV(현재 선택 기준, 만원)", None)
    set_cell(ws, "B4", None, formula=f"=SUM(검토표!$T$2:$T${last})", number_format=MANWON_FMT)
    set_cell(ws, "A5", "포트폴리오 LGD(현재 선택 기준)", None)
    set_cell(ws, "B5", None, formula="=1-B4/B3", number_format=PCT_FMT)

    set_cell(ws, "A7", "참고(고정값, results/exp009_recovery.md 표1 그대로 옮김) — 기본 낙찰가율, 보증금×매각시점", None)
    set_cell(ws, "A8", "보증금\\개월", None)
    for j, m in enumerate([12, 24, 36], start=2):
        from openpyxl.utils import get_column_letter
        set_cell(ws, f"{get_column_letter(j)}8", m, None)
    table1 = {
        "없음": [0.155854, 0.218383, 0.276281],
        "기본": [0.865485, 0.875449, 0.884675],
        "깡통": [1.0, 1.0, 1.0],
    }
    for i, (dep, vals) in enumerate(table1.items(), start=9):
        set_cell(ws, f"A{i}", dep, None)
        for j, v in enumerate(vals, start=2):
            from openpyxl.utils import get_column_letter
            set_cell(ws, f"{get_column_letter(j)}{i}", v, None, PCT_FMT)
    set_cell(ws, "A13", "위 참고표는 가정 시트 드롭다운을 바꿔도 갱신되지 않는다(고정값). B5(포트폴리오 LGD)만 실시간이다.", None)

    for col, w in [("A", 55), ("B", 14), ("C", 14), ("D", 14)]:
        ws.column_dimensions[col].width = w
    return ws


def main():
    port = pd.read_parquet(PORTFOLIO_V2_PARQUET)
    wb = Workbook()
    wb.remove(wb.active)
    build_assumptions(wb)
    _, n_rows = build_review(wb, port)
    build_summary(wb, n_rows)
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_XLSX)
    size_kb = OUT_XLSX.stat().st_size / 1024
    print(f"저장: {OUT_XLSX} ({size_kb:.1f}KB, {len(wb.sheetnames)}시트: {wb.sheetnames})")

    # 수식 오류 점검
    n_ref_err = 0
    n_formula = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    n_formula += 1
                    if "#REF!" in cell.value:
                        n_ref_err += 1
    print(f"수식 셀 {n_formula}개 중 #REF! {n_ref_err}건")
    return n_ref_err, n_formula


if __name__ == "__main__":
    main()
