"""EXP-007 단계 9: results/npl_portfolio_review.xlsx 생성(openpyxl, 계산 셀은 수식).

시트 8개: 요약, 가정, 검토표, 배당표, 비교사례, 민감도, 모델성능, 출처·한계.
모든 금액은 만원 단위로 통일한다(원본은 원 단위이므로 10,000으로 나눠 입력 값으로 쓴다).
입력 셀은 파란 글씨, 같은 시트 안 수식 셀은 검정, 다른 시트를 참조하는 수식 셀은 초록으로 칠한다.
민감도·요약 시트의 보조 계산은 숨김 시트(_계산)에 "같은 행만 참조하는 짧은 수식"으로 풀어 쓴다
(한 셀에 전체 배당 단계를 중첩하면 수식이 수천 자로 길어져 위험하므로 피한다).
"""

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent
PORTFOLIO_PARQUET = ROOT / "data" / "processed" / "portfolio_v1.parquet"
COMPS_PARQUET = ROOT / "data" / "processed" / "portfolio_comps_v1.parquet"
TWO_PATH_JSON = ROOT / "results" / "exp007_two_path_mdape.json"
OUT_XLSX = ROOT / "results" / "npl_portfolio_review.xlsx"

BLUE = Font(color="0070C0")
BLACK = Font(color="000000")
GREEN = Font(color="008000")
BOLD = Font(bold=True)
MANWON_FMT = '#,##0'
PCT_FMT = '0.00%'
N_ITEMS = 20
N_SOLD = 12


def set_cell(ws, addr, value, kind="value", formula=None, number_format=None):
    cell = ws[addr]
    if formula is not None:
        cell.value = formula
        cell.font = GREEN if "!" in formula else BLACK
    else:
        cell.value = value
        if kind == "input":
            cell.font = BLUE
        elif kind == "header":
            cell.font = BOLD
    if number_format:
        cell.number_format = number_format
    return cell


def add_list_validation(ws, addr, options_or_range):
    if options_or_range.startswith("="):
        dv = DataValidation(type="list", formula1=options_or_range[1:], allow_blank=False)
    else:
        dv = DataValidation(type="list", formula1=f'"{options_or_range}"', allow_blank=False)
    ws.add_data_validation(dv)
    dv.add(ws[addr])


# ---------------------------------------------------------------- 가정 ----
def build_assumptions_sheet(wb):
    ws = wb.create_sheet("가정")
    set_cell(ws, "A1", "가정값 (모두 이 시트에서 바꿀 수 있다)", "header")

    rows = [
        ("LTV", 0.70, "사용자 결정 2026-11-04(가상값) — 추정 시세 기준 대출원금 비율", None),
        ("채권최고액_비율", 1.20, "사용자 결정(가상값) — 대출원금의 120%", None),
        ("약정금리", 0.12, "사용자 결정(가상값) — 연 12%", None),
        ("연체가산", 0.03, "사용자 결정(가상값) — 약정금리 + 3%p", None),
        ("연체금리(계산)", None, "약정금리+연체가산(법령 원문 확인 필요로 표시)", "=B4+B5"),
        ("연체시작_개월전", 12, "사용자 결정(가상값) — 최초 개찰일 12개월 전부터 연체", None),
        ("집행비용율", 0.01, "사용자 결정(가상값) — 매각가의 1%", None),
        ("당해세_만원", 0, "사용자 결정 — 이번 실험 범위 밖(0원 가정)", None),
        ("할인율", 0.08, "사용자 결정(가상값) — 연 8%", None),
        ("최우선변제_보증금한도_만원", 16500, "서울 소액임차인 최우선변제 현행 기준표(법령 원문 확인 필요)", None),
        ("최우선변제_최대지급_만원", 5500, "서울 소액임차인 최우선변제 현행 기준표(법령 원문 확인 필요)", None),
        ("최우선변제_매각가비율한도", 0.5, "사용자 결정 — 최우선변제 합계는 매각가의 1/2 한도", None),
        ("근저당_설정일", "2023-03-01", "사용자 결정(가상값) — 1순위 근저당, 최우선변제 현행 기준표 적용", None),
    ]
    for i, (name, value, note, formula) in enumerate(rows, start=2):
        set_cell(ws, f"A{i}", name)
        if formula:
            set_cell(ws, f"B{i}", None, formula=formula)
        else:
            set_cell(ws, f"B{i}", value, "input")
        set_cell(ws, f"C{i}", note)

    set_cell(ws, "A16", "선택(드롭다운)", "header")
    set_cell(ws, "A17", "매각시점시나리오"); set_cell(ws, "B17", "기본", "input")
    set_cell(ws, "A18", "매각가경로"); set_cell(ws, "B18", "시세", "input")
    set_cell(ws, "A19", "보증금시나리오"); set_cell(ws, "B19", "기본", "input")
    set_cell(ws, "A20", "시나리오개월수(계산)")
    set_cell(ws, "B20", None, formula="=VLOOKUP(B17,D22:E24,2,FALSE)")

    set_cell(ws, "D21", "시나리오"); set_cell(ws, "E21", "개월수")
    for i, (name, months) in enumerate([("낙관", 12), ("기본", 24), ("비관", 36)], start=22):
        set_cell(ws, f"D{i}", name); set_cell(ws, f"E{i}", months)

    add_list_validation(ws, "B17", "낙관,기본,비관")
    add_list_validation(ws, "B18", "시세,감정가")
    add_list_validation(ws, "B19", "없음,기본,깡통")

    for col, w in [("A", 26), ("B", 14), ("C", 60), ("D", 10), ("E", 10)]:
        ws.column_dimensions[col].width = w
    return ws


# ---------------------------------------------------------------- 검토표 ----
def build_review_sheet(wb, portfolio):
    ws = wb.create_sheet("검토표")
    headers = [
        "cltrMngNo", "masked_주소", "결과구분", "극단물건", "구", "건물면적_m2",
        "감정가_만원", "추정시세_만원", "rho_mkt_est_%", "rho_appr_est_%",
        "대출원금_EAD_만원", "보증금_없음_만원", "보증금_기본_만원", "보증금_깡통_만원",
        "선택보증금_만원", "실제낙찰가_만원", "예상매각가_만원", "연체연수", "청구금액_만원",
        "집행비용_만원", "남은1_만원", "최우선변제후보_만원", "최우선변제상한_만원", "최우선변제_만원",
        "남은2_만원", "당해세지급_만원", "남은3_만원", "보증금잔액_만원", "보증금잔액지급_만원",
        "남은4_본채권전_만원", "채권최고액_만원", "R_회수액_만원", "PV_만원", "LGD",
    ]
    for j, h in enumerate(headers, start=1):
        set_cell(ws, f"{get_column_letter(j)}1", h, "header")

    for i, (_, row) in enumerate(portfolio.iterrows(), start=2):
        set_cell(ws, f"A{i}", row["cltrMngNo"], "input")
        set_cell(ws, f"B{i}", row["masked_주소"], "input")
        set_cell(ws, f"C{i}", row["결과구분"], "input")
        set_cell(ws, f"D{i}", bool(row["극단물건"]), "input")
        set_cell(ws, f"E{i}", row["구"], "input")
        set_cell(ws, f"F{i}", round(row["건물면적"], 2), "input")
        set_cell(ws, f"G{i}", round(row["감정가"] / 10000, 1) if pd.notna(row["감정가"]) else None, "input",
                  number_format=MANWON_FMT)
        set_cell(ws, f"H{i}", round(row["market_price"] / 10000, 1), "input", number_format=MANWON_FMT)
        set_cell(ws, f"I{i}", round(row["rho_mkt_est"], 2), "input")
        set_cell(ws, f"J{i}", round(row["rho_appr_est"], 2), "input")
        set_cell(ws, f"K{i}", None, formula=f"=H{i}*가정!$B$2", number_format=MANWON_FMT)
        set_cell(ws, f"L{i}", 0, "input", number_format=MANWON_FMT)
        set_cell(ws, f"M{i}", round(row["보증금_기본"] / 10000, 1), "input", number_format=MANWON_FMT)
        set_cell(ws, f"N{i}", None, formula=f"=H{i}*0.9", number_format=MANWON_FMT)
        set_cell(ws, f"O{i}", None,
                  formula=f'=CHOOSE(MATCH(가정!$B$19,{{"없음","기본","깡통"}},0),L{i},M{i},N{i})',
                  number_format=MANWON_FMT)
        set_cell(ws, f"P{i}", round(row["낙찰가"] / 10000, 1) if pd.notna(row["낙찰가"]) else None, "input",
                  number_format=MANWON_FMT)
        set_cell(ws, f"Q{i}", None,
                  formula=f'=IFERROR(IF(가정!$B$18="시세",H{i}*I{i}/100,IF(G{i}="","",G{i}*J{i}/100)),"")',
                  number_format=MANWON_FMT)
        set_cell(ws, f"R{i}", None, formula=f"=(가정!$B$7+가정!$B$20)/12")
        set_cell(ws, f"S{i}", None, formula=f"=K{i}*(1+가정!$B$6*R{i})", number_format=MANWON_FMT)
        set_cell(ws, f"T{i}", None, formula=f'=IFERROR(Q{i}*가정!$B$8,"")', number_format=MANWON_FMT)
        set_cell(ws, f"U{i}", None, formula=f'=IFERROR(MAX(Q{i}-T{i},0),"")', number_format=MANWON_FMT)
        set_cell(ws, f"V{i}", None,
                  formula=f'=IFERROR(IF(O{i}<=가정!$B$11,MIN(O{i},가정!$B$12),0),"")', number_format=MANWON_FMT)
        set_cell(ws, f"W{i}", None, formula=f'=IFERROR(Q{i}*가정!$B$13,"")', number_format=MANWON_FMT)
        set_cell(ws, f"X{i}", None, formula=f'=IFERROR(MAX(MIN(V{i},W{i},U{i}),0),"")', number_format=MANWON_FMT)
        set_cell(ws, f"Y{i}", None, formula=f'=IFERROR(MAX(U{i}-X{i},0),"")', number_format=MANWON_FMT)
        set_cell(ws, f"Z{i}", None, formula=f"=IFERROR(MIN(가정!$B$9,Y{i}),\"\")", number_format=MANWON_FMT)
        set_cell(ws, f"AA{i}", None, formula=f'=IFERROR(MAX(Y{i}-Z{i},0),"")', number_format=MANWON_FMT)
        set_cell(ws, f"AB{i}", None, formula=f'=IFERROR(MAX(O{i}-X{i},0),"")', number_format=MANWON_FMT)
        set_cell(ws, f"AC{i}", None, formula=f'=IFERROR(MAX(MIN(AB{i},AA{i}),0),"")', number_format=MANWON_FMT)
        set_cell(ws, f"AD{i}", None, formula=f'=IFERROR(MAX(AA{i}-AC{i},0),"")', number_format=MANWON_FMT)
        set_cell(ws, f"AE{i}", None, formula=f"=K{i}*가정!$B$3", number_format=MANWON_FMT)
        set_cell(ws, f"AF{i}", None, formula=f'=IFERROR(MAX(MIN(AE{i},S{i},AD{i}),0),"")', number_format=MANWON_FMT)
        set_cell(ws, f"AG{i}", None, formula=f'=IFERROR(AF{i}/(1+가정!$B$10)^(가정!$B$20/12),"")', number_format=MANWON_FMT)
        set_cell(ws, f"AH{i}", None, formula=f'=IFERROR(IF(K{i}=0,"",1-AG{i}/K{i}),"")', number_format=PCT_FMT)

    last = N_ITEMS + 1
    lgd_col = "AH"
    rule = CellIsRule(operator="greaterThanOrEqual", formula=["0.5"], fill=None)
    from openpyxl.styles import PatternFill
    rule.dxf = None
    ws.conditional_formatting.add(
        f"{lgd_col}2:{lgd_col}{last}",
        CellIsRule(operator="greaterThanOrEqual", formula=["0.5"],
                   fill=PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid"))
    )

    for col in ["A", "B"]:
        ws.column_dimensions[col].width = 20
    return ws


# ---------------------------------------------------------------- 배당표 ----
def build_waterfall_sheet(wb):
    ws = wb.create_sheet("배당표")
    set_cell(ws, "A1", "물건 선택(masked_주소)", "header")
    set_cell(ws, "B1", "=검토표!B2", "input")
    add_list_validation(ws, "B1", "=검토표!$B$2:$B$21")

    steps = [
        ("매각가(선택 경로)", "Q"), ("집행비용", "T"), ("최우선변제", "X"), ("당해세", "Z"),
        ("선순위보증금잔액 지급", "AC"), ("남은금액(본채권 배당 전)", "AD"), ("채권최고액", "AE"),
        ("청구금액", "S"), ("본채권 회수액 R", "AF"), ("PV(현재가치)", "AG"), ("LGD", "AH"),
    ]
    set_cell(ws, "A3", "배당 단계", "header")
    set_cell(ws, "B3", "금액(만원) / 비율", "header")
    for i, (label, col) in enumerate(steps, start=4):
        set_cell(ws, f"A{i}", label)
        formula = f'=INDEX(검토표!${col}$2:${col}$21,MATCH($B$1,검토표!$B$2:$B$21,0))'
        number_format = PCT_FMT if col == "AH" else MANWON_FMT
        set_cell(ws, f"B{i}", None, formula=formula, number_format=number_format)

    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 18
    return ws


# ---------------------------------------------------------------- 비교사례 ----
def build_comps_sheet(wb, portfolio, comps):
    ws = wb.create_sheet("비교사례")
    addr_map = portfolio.set_index("cltrMngNo")["masked_주소"]
    comps = comps.copy()
    comps["masked_주소"] = comps["cltrMngNo"].map(addr_map)
    cols = ["masked_주소", "순위", "법정동", "면적", "연식", "층", "㎡당가격_만원", "계약월", "거리_km"]
    comps = comps[cols]

    for j, h in enumerate(cols, start=1):
        set_cell(ws, f"{get_column_letter(j)}1", h, "header")
    for i, (_, row) in enumerate(comps.iterrows(), start=2):
        for j, c in enumerate(cols, start=1):
            val = row[c]
            if hasattr(val, "item"):
                val = val.item()
            set_cell(ws, f"{get_column_letter(j)}{i}", val, "input")

    ws.column_dimensions["A"].width = 20
    return ws


# ---------------------------------------------------------------- _계산(숨김) ----
SENS_SHOCKS = [-10, -5, 0, 5, 10]
SENS_MONTHS = [("낙관", 12), ("기본", 24), ("비관", 36)]
DEPOSIT_COLS = {"없음": "L", "기본": "M", "깡통": "N"}


def build_hidden_calc_sheet(wb):
    ws = wb.create_sheet("_계산")
    ws.sheet_state = "hidden"

    headers = ["검토표행", "조합", "개월수", "rho_shock", "매각가", "집행비용", "남은1", "보증금선택",
               "최우선변제후보", "최우선변제상한", "최우선변제", "남은2", "당해세", "남은3",
               "보증금잔액", "보증금잔액지급", "남은4", "대출원금", "채권최고액", "청구금액", "R", "PV"]
    for j, h in enumerate(headers, start=1):
        set_cell(ws, f"{get_column_letter(j)}1", h, "header")

    r = 2
    blocks = {}  # combo_label -> (start_row, end_row)
    for month_name, months in SENS_MONTHS:
        for shock in SENS_SHOCKS:
            label = f"{month_name}_{shock:+d}"
            start = r
            for item_row in range(2, 2 + N_ITEMS):
                write_calc_row(ws, r, item_row, months, shock, deposit_col=None)
                ws[f"B{r}"] = label
                r += 1
            blocks[label] = (start, r - 1)

    # 요약 시트용: 보증금 시나리오 3개(현재 선택 매각시점시나리오·개월수, shock=0, deposit=L/M/N 직접)
    for dep_name, dep_col in DEPOSIT_COLS.items():
        label = f"dep_{dep_name}"
        start = r
        for item_row in range(2, 2 + N_ITEMS):
            write_calc_row(ws, r, item_row, months="가정!$B$20", shock=0, deposit_col=dep_col)
            ws[f"B{r}"] = label
            r += 1
        blocks[label] = (start, r - 1)

    return ws, blocks


def write_calc_row(ws, r, item_row, months, shock, deposit_col):
    set_cell(ws, f"A{r}", f"검토표!{item_row}")
    if isinstance(months, str):
        set_cell(ws, f"C{r}", None, formula=f"={months}")
    else:
        set_cell(ws, f"C{r}", months)
    set_cell(ws, f"D{r}", shock)

    sale = (f'IF(가정!$B$18="시세",검토표!$H${item_row}*(검토표!$I${item_row}+D{r})/100,'
            f'IF(검토표!$G${item_row}="","",검토표!$G${item_row}*(검토표!$J${item_row}+D{r})/100))')
    set_cell(ws, f"E{r}", None, formula=f"=IFERROR({sale},\"\")")
    set_cell(ws, f"F{r}", None, formula=f'=IFERROR(E{r}*가정!$B$8,"")')
    set_cell(ws, f"G{r}", None, formula=f'=IFERROR(MAX(E{r}-F{r},0),"")')

    if deposit_col is None:
        deposit_ref = f"검토표!$O${item_row}"
    else:
        deposit_ref = f"검토표!${deposit_col}${item_row}"
    set_cell(ws, f"H{r}", None, formula=f"={deposit_ref}")
    set_cell(ws, f"I{r}", None, formula=f'=IFERROR(IF(H{r}<=가정!$B$11,MIN(H{r},가정!$B$12),0),"")')
    set_cell(ws, f"J{r}", None, formula=f'=IFERROR(E{r}*가정!$B$13,"")')
    set_cell(ws, f"K{r}", None, formula=f'=IFERROR(MAX(MIN(I{r},J{r},G{r}),0),"")')
    set_cell(ws, f"L{r}", None, formula=f'=IFERROR(MAX(G{r}-K{r},0),"")')
    set_cell(ws, f"M{r}", None, formula=f'=IFERROR(MIN(가정!$B$9,L{r}),"")')
    set_cell(ws, f"N{r}", None, formula=f'=IFERROR(MAX(L{r}-M{r},0),"")')
    set_cell(ws, f"O{r}", None, formula=f'=IFERROR(MAX(H{r}-K{r},0),"")')
    set_cell(ws, f"P{r}", None, formula=f'=IFERROR(MAX(MIN(O{r},N{r}),0),"")')
    set_cell(ws, f"Q{r}", None, formula=f'=IFERROR(MAX(N{r}-P{r},0),"")')
    set_cell(ws, f"R{r}", None, formula=f"=검토표!$K${item_row}")
    set_cell(ws, f"S{r}", None, formula=f"=R{r}*가정!$B$3")
    set_cell(ws, f"T{r}", None, formula=f"=R{r}*(1+가정!$B$6*((가정!$B$7+C{r})/12))")
    set_cell(ws, f"U{r}", None, formula=f'=IFERROR(MAX(MIN(S{r},T{r},Q{r}),0),"")')
    set_cell(ws, f"V{r}", None, formula=f'=IFERROR(U{r}/(1+가정!$B$10)^(C{r}/12),0)')


# ---------------------------------------------------------------- 민감도 ----
def build_sensitivity_sheet(wb, blocks):
    ws = wb.create_sheet("민감도")
    set_cell(ws, "A1", "낙찰가율 민감도 × 매각 시점 — 포트폴리오 PV 합계(만원, 만원단위, 현재 선택 경로·보증금 기준)", "header")
    set_cell(ws, "A2", "매각시점\\낙찰가율 변동(%p)", "header")
    for j, shock in enumerate(SENS_SHOCKS, start=2):
        set_cell(ws, f"{get_column_letter(j)}2", f"{shock:+d}%p", "header")

    for i, (month_name, months) in enumerate(SENS_MONTHS, start=3):
        set_cell(ws, f"A{i}", month_name, "header")
        for j, shock in enumerate(SENS_SHOCKS, start=2):
            label = f"{month_name}_{shock:+d}"
            start, end = blocks[label]
            set_cell(ws, f"{get_column_letter(j)}{i}", None,
                      formula=f"=SUM(_계산!$V${start}:$V${end})", number_format=MANWON_FMT)

    for col in ["A", "B", "C", "D", "E", "F"]:
        ws.column_dimensions[col].width = 16
    return ws


# ---------------------------------------------------------------- 모델성능 ----
def build_model_perf_sheet(wb, two_path):
    ws = wb.create_sheet("모델성능")
    set_cell(ws, "A1", "EXP-002~004 시세 MdAPE (results/*.md에서 그대로 옮김, 정적 값)", "header")
    rows1 = [
        ("B0(중앙값 기준선)", 0.2452, "results/exp002_baseline.md"),
        ("B1(셀 기반 기준선, 공식 채택)", 0.1704, "results/exp002_baseline.md"),
        ("FAISS k-NN(EXP-003)", 0.1458, "results/exp003_faiss.md"),
        ("FAISS k-NN(EXP-004, 최종 채택)", 0.1335, "results/exp004_faiss_ext.md"),
    ]
    for i, (name, val, src) in enumerate(rows1, start=2):
        set_cell(ws, f"A{i}", name); set_cell(ws, f"B{i}", val, number_format=PCT_FMT); set_cell(ws, f"C{i}", src)

    set_cell(ws, "A7", "EXP-005 정정 LOO MdAE(v2, %p)", "header")
    set_cell(ws, "A8", "rho_appr"); set_cell(ws, "B8", 6.67, number_format="0.00")
    set_cell(ws, "A9", "rho_mkt"); set_cell(ws, "B9", 14.00, number_format="0.00")
    set_cell(ws, "C8", "results/exp005_auction.md"); set_cell(ws, "C9", "results/exp005_auction.md")

    set_cell(ws, "A11", "EXP-007 v3 재적합 LOO MdAE(%p, 정적 값)", "header")
    set_cell(ws, "A12", "rho_appr"); set_cell(ws, "B12", 6.67, number_format="0.00")
    set_cell(ws, "A13", "rho_mkt"); set_cell(ws, "B13", 12.87, number_format="0.00")
    set_cell(ws, "C12", "src/rho_refit_v3.py 콘솔 출력(로컬)"); set_cell(ws, "C13", "src/rho_refit_v3.py 콘솔 출력(로컬)")

    set_cell(ws, "A15", "두 경로 비교 MdAPE(88건, 판정 없음, 기록만, 정적 값)", "header")
    set_cell(ws, "A16", "시세 경로"); set_cell(ws, "B16", two_path["mdape_시세경로"], number_format=PCT_FMT)
    set_cell(ws, "A17", "감정가 경로"); set_cell(ws, "B17", two_path["mdape_감정가경로"], number_format=PCT_FMT)
    set_cell(ws, "C16", "results/exp007_two_path_mdape.json"); set_cell(ws, "C17", "results/exp007_two_path_mdape.json")

    set_cell(ws, "A19", "낙찰 12건 사후 검증(현재 선택 경로·시나리오, 수식, 검토표 참조)", "header")
    headers = ["masked_주소", "실제낙찰가_만원", "예측매각가_만원", "APE"]
    for j, h in enumerate(headers, start=1):
        set_cell(ws, f"{get_column_letter(j)}20", h, "header")
    for i, row_i in enumerate(range(2, 2 + N_SOLD), start=21):
        set_cell(ws, f"A{i}", None, formula=f"=검토표!B{row_i}")
        set_cell(ws, f"B{i}", None, formula=f"=검토표!P{row_i}", number_format=MANWON_FMT)
        set_cell(ws, f"C{i}", None, formula=f"=검토표!Q{row_i}", number_format=MANWON_FMT)
        set_cell(ws, f"D{i}", None, formula=f'=IFERROR(ABS(B{i}-C{i})/B{i},"")', number_format=PCT_FMT)
    last = 20 + N_SOLD
    set_cell(ws, f"A{last + 1}", "MdAPE(중앙값)", "header")
    set_cell(ws, f"D{last + 1}", None, formula=f"=MEDIAN(D21:D{last})", number_format=PCT_FMT)

    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["C"].width = 32
    return ws, f"D{last + 1}"


# ---------------------------------------------------------------- 출처·한계 ----
def build_sources_sheet(wb):
    ws = wb.create_sheet("출처·한계")
    lines = [
        ("데이터 출처", ""),
        ("DS-01 국토교통부 연립다세대 매매 실거래가", "공공데이터포털 부동산 실거래가 API, 수집 2024-11~2026-10"),
        ("DS-02 온비드 물건 입찰결과목록(서울 주거용)", "온비드 공매 API, 수집 2024-11~2026-10"),
        ("DS-03 국토교통부 연립다세대 전월세 실거래가", "공공데이터포털 전월세 실거래가 API, 신규 호출 624건"),
        ("건축HUB 건축물대장 표제부", "국토교통부 건축HUB API(SYN-007)"),
        ("", ""),
        ("가상값 목록(실제 채권 조건 아님)", ""),
        ("LTV 70%, 채권최고액 120%, 약정금리 12%, 연체가산 3%p", "가정 시트 참조, 전부 사용자 결정(2026-11-04)"),
        ("연체시작=최초개찰일 12개월 전, 집행비용=매각가 1%, 당해세 0원, 할인율 8%", "가정 시트 참조"),
        ("", ""),
        ("범위 밖 규칙(이번 실험에서 다루지 않음)", ""),
        ("흡수배당", ""), ("임금채권", ""), ("가압류 안분배당", ""),
        ("당해세 예외(2023-04-01) 세부", ""), ("상가 환산보증금", ""),
        ("", ""),
        ("공매 ≠ 법원경매", "온비드 공매 데이터이며 법원경매와 절차·통계가 다르다(EXP-005 주의 참조)"),
        ("", ""),
        ("보증금 측정 한계(EXP-006)", ""),
        ("전월세 수집 기간이 원 임대차 계약 시점보다 늦을 수 있음", "results/exp006_mixture.md 해석 참조"),
        ("전세·월세 보증금 혼재", "results/exp006_mixture.md 해석 참조"),
        ("인수부담 의심 집단 보증금비율 유효 6건(표본 작음)", "results/exp006_mixture.md 표5 참조"),
        ("EXP-006 가설(혼합 구조) 자체는 미지지로 종료됨 — 이번 실험은 EXP-005 단일 M2 유지 전제로 진행", "results/exp006_mixture.md 판정 참조"),
    ]
    for i, (a, b) in enumerate(lines, start=1):
        bold = a and not b and a not in ("",) and i != 1
        set_cell(ws, f"A{i}", a, "header" if (a and (b == "" and a != "")) else "value")
        set_cell(ws, f"B{i}", b)
    ws.column_dimensions["A"].width = 55
    ws.column_dimensions["B"].width = 60
    return ws


# ---------------------------------------------------------------- 요약 ----
def build_summary_sheet(wb, blocks):
    ws = wb.create_sheet("요약", 0)
    set_cell(ws, "A1", "EXP-007 포트폴리오 회수액·LGD 요약(모두 수식, 가정 시트의 선택을 따른다)", "header")

    set_cell(ws, "A3", "20건 합계 EAD(대출원금, 만원)")
    set_cell(ws, "B3", None, formula="=SUM(검토표!$K$2:$K$21)", number_format=MANWON_FMT)

    set_cell(ws, "A5", "시나리오별 예상 회수액 PV 합계(만원) — 현재 선택 경로·보증금 기준", "header")
    for i, (month_name, months) in enumerate(SENS_MONTHS, start=6):
        label = f"{month_name}_{0:+d}"
        start, end = blocks[label]
        set_cell(ws, f"A{i}", month_name)
        set_cell(ws, f"B{i}", None, formula=f"=SUM(_계산!$V${start}:$V${end})", number_format=MANWON_FMT)

    set_cell(ws, "A10", "포트폴리오 LGD(현재 선택 경로·매각시점·보증금 전체 기준)")
    set_cell(ws, "B10", None, formula="=1-SUM(검토표!$AG$2:$AG$21)/SUM(검토표!$K$2:$K$21)", number_format=PCT_FMT)

    set_cell(ws, "A12", "보증금 시나리오별 LGD(현재 선택 경로·매각시점 기준)", "header")
    for i, dep_name in enumerate(["없음", "기본", "깡통"], start=13):
        label = f"dep_{dep_name}"
        start, end = blocks[label]
        set_cell(ws, f"A{i}", dep_name)
        set_cell(ws, f"B{i}", None,
                  formula=f"=1-SUM(_계산!$V${start}:$V${end})/SUM(검토표!$K$2:$K$21)", number_format=PCT_FMT)

    set_cell(ws, "A17", "사후 검증 MdAPE(낙찰 12건, 현재 선택 경로·시나리오 기준)")
    set_cell(ws, "B17", None, formula="=모델성능!D33", number_format=PCT_FMT)

    for cell in ["B10", "B13", "B14", "B15", "B17"]:
        from openpyxl.styles import PatternFill
        ws.conditional_formatting.add(
            cell, CellIsRule(operator="greaterThanOrEqual", formula=["0.5"],
                              fill=PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid"))
        )

    ws.column_dimensions["A"].width = 55
    ws.column_dimensions["B"].width = 16

    ws.print_area = "A1:B18"
    ws.page_setup.orientation = "portrait"
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToPage = True
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    return ws


def main():
    portfolio = pd.read_parquet(PORTFOLIO_PARQUET)
    portfolio = pd.concat([
        portfolio[portfolio["결과구분"] == "낙찰"], portfolio[portfolio["결과구분"] == "미낙찰"],
    ]).reset_index(drop=True)
    comps = pd.read_parquet(COMPS_PARQUET)
    with open(TWO_PATH_JSON, encoding="utf-8") as f:
        two_path = json.load(f)

    wb = Workbook()
    wb.remove(wb.active)

    build_assumptions_sheet(wb)
    build_review_sheet(wb, portfolio)
    build_waterfall_sheet(wb)
    build_comps_sheet(wb, portfolio, comps)
    _, blocks = build_hidden_calc_sheet(wb)
    build_sensitivity_sheet(wb, blocks)
    build_model_perf_sheet(wb, two_path)
    build_sources_sheet(wb)
    build_summary_sheet(wb, blocks)

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_XLSX)
    size_mb = OUT_XLSX.stat().st_size / 1024 / 1024
    print(f"저장: {OUT_XLSX} ({size_mb:.2f} MB)")
    if size_mb > 1.0:
        raise RuntimeError(f"엑셀 파일이 1MB를 넘음({size_mb:.2f}MB) — 멈춤 지점")
    return OUT_XLSX


if __name__ == "__main__":
    main()
