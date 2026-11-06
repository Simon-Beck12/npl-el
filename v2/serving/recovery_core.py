"""회수액 6줄 계산 — v2/recovery.py `compute_recovery_v2`와 같은 식을 서빙용으로 옮긴 사본.

서빙 이미지는 faiss·학습 데이터 로더를 넣지 않으려고 v2/recovery.py를 import하지 않는다.
두 함수가 같은 값을 내는지는 tests/test_recovery_parity.py가 무작위 입력 2,000건으로 확인한다.
식이나 상수를 바꾸면 v2/recovery.py와 이 파일을 함께 바꾼다.
"""

PRIORITY_CAP = 165_000_000   # 최우선변제 대상 보증금 상한(원)
PRIORITY_MAX = 55_000_000    # 최우선변제 금액 상한(원)
DISCOUNT_RATE = 0.08
LTV = 0.70                   # EAD 미입력 시 가정값(v2와 동일): EAD = 시세 × 0.70


def compute_recovery(market_price, rho, deposit, ead, months, discount_rate=DISCOUNT_RATE):
    s = market_price * rho                                     # ① 예상 매각가
    k = s * 0.01                                               # ② 집행비용
    if deposit <= PRIORITY_CAP:
        p = min(deposit, PRIORITY_MAX, s * 0.5)                 # ③ 최우선변제
    else:
        p = 0.0
    d2 = deposit - p                                            # ④ 보증금 잔액
    r = max(0.0, min(ead, s - k - p - d2))                      # ⑤ 회수액
    pv = r / ((1 + discount_rate) ** (months / 12))             # ⑥ 현재가치
    lgd = (1 - pv / ead) if ead > 0 else None
    return {"S": s, "K": k, "P": p, "D2": d2, "R": r, "PV": pv, "EAD": ead, "LGD": lgd}
