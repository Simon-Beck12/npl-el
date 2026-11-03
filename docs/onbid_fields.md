# 온비드 물건 입찰결과목록 API 필드 (DS-02)

출처: `data/raw/onbid/202411_p1.json` (첫 응답). 경로: `body.items.item[*]`.

| 필드명 | 예시 값 | 비고 |
|---|---|---|
| cltrMngNo | "2023-0500-023714" | 물건관리번호 (물건 단위 키) |
| pbctCdtnNo | 5062152 | 공매조건번호 |
| onbidCltrno | 1670969 | 온비드 물건번호 |
| onbidPbancNo | 796025 | 온비드 공고번호 |
| pbctNo | 9873498 | 공매번호 |
| prptDivCd | "0005" | 재산구분 코드 |
| prptDivNm | "기타일반재산" | 재산구분명 |
| dspsMthodCd | "0001" | 처분방식 코드(매각) |
| dspsMthodNm | "매각" | 처분방식명 |
| cltrUsgLclsCtgrId | "10000" | 용도 대분류 코드 |
| cltrUsgMclsCtgrId | "10200" | 용도 중분류 코드(주거용건물) |
| cltrUsgSclsCtgrId | "10208" | 용도 소분류 코드 |
| cltrUsgLclsCtgrNm | "부동산" | 용도 대분류명 |
| cltrUsgMclsCtgrNm | "주거용건물" | 용도 중분류명 |
| cltrUsgSclsCtgrNm | "도시형생활주택" | 용도 소분류명 |
| onbidCltrNm | "서울시 중구 신당동 251-73 1001호 도시형생활주택" | 물건명(주소 포함, 파싱 대상) |
| landSqms | 0 | 토지면적(㎡) |
| bldSqms | 14.54 | 건물면적(㎡) |
| pbctNsq | "3" | 공매 회차 |
| pbctsn | "1" | 공매 차수 |
| apslEvlAmt | 268000000 | 감정가 |
| lowstBidPrcIndctCont | "370000000" | 최저입찰가(문자열, 숫자 변환 필요) |
| apslPrcCtrsLowstBidRto | None | 감정가 대비 최저입찰가 비율(결측 가능) |
| frstCtrsLowstBidPrcRto | None | 최초가 대비 최저입찰가 비율(결측 가능) |
| feeRate | None | 수수료율(결측 가능) |
| cltrOpbdDt | "202411301000" | 개찰일시(YYYYMMDDHHmm) |
| pbctStatCd | "0011" | 공매상태 코드 |
| pbctStatNm | "유찰" | 공매상태명 |
| scfbAmt | None | 낙찰가(성립금액). 유찰·취소 시 None, 낙찰 시 문자열 숫자 |
| vldBddrNope | 0 | 유효 입찰자 수 |
| nfctBddrNope | 0 | 무효 입찰자 수 |
| apslPrcCtrsScfbPrcRto | 0 | 낙찰가율(감정가 대비, API 제공값). 유찰 시 0 |
| lowstBidCtrsScfbPrcRto | 0 | 낙찰가÷최저입찰가 비율(API 제공값). 유찰 시 0 |
| drvDstc | None | 차량 관련(주거용 아님, 결측) |
| cltrMkrNm | None | 차량 관련(주거용 아님, 결측) |
| carMdlNm | None | 차량 관련(주거용 아님, 결측) |
| yrmdl | None | 차량 관련(주거용 아님, 결측) |
| fuelCont | None | 차량 관련(주거용 아님, 결측) |
| dsvlm | None | 차량 관련(주거용 아님, 결측) |

## pbctStatCd 분포 (전체 24개월, 16,283회차 행 기준)

| 코드 | 명칭 | 건수 |
|---|---|---|
| 0011 | 유찰 | 13,779 |
| 0012 | 취소 | 1,346 |
| 0009 | 수의계약가능 | 709 |
| 0010 | 낙찰 | 436 |
| 0003 | 입찰마감 | 12 |
| 0006 | 개찰중 | 1 |

주의: 공매 데이터이며 법원경매와 다르다.
