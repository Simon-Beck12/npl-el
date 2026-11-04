# 연립다세대 전월세 실거래 응답 필드 — 첫 응답 예시

출처: RH_RENT_URL 샘플 호출(LAWD_CD=11680 강남구, DEAL_YMD=202501, 1페이지, 첫 item). data/raw/rh_rent/ 수집 전 사전 확인(EXP-006 9~10단계).

| 필드명 | 예시 값 |
|---|---|
| buildYear | 1993 |
| contractTerm | (공백) |
| contractType | (공백) |
| dealDay | 4 |
| dealMonth | 1 |
| dealYear | 2025 |
| deposit | 58,000 |
| excluUseAr | 59.75 |
| floor | 1 |
| houseType | 연립 |
| jibun | 739 |
| mhouseNm | 청솔빌리지 |
| monthlyRent | 0 |
| preDeposit | (공백) |
| preMonthlyRent | (공백) |
| sggCd | 11680 |
| umdNm | 일원동 |
| useRRRight | (공백) |

비고: 매매(DS-01, `docs/fields.md`)와 달리 `cdealType`/`cdealDay`(계약 해제 표시) 필드가 전월세 응답에는 존재하지 않는다(같은 LAWD_CD·DEAL_YMD 응답 100건 전체 확인, 필드 구성 전부 동일 — 해제 표시 필드 자체가 없음). 따라서 "계약 해제 표시 제거" 단계는 해당 없음(0건)으로 처리한다.
