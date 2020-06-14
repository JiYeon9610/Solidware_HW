# solidware_MachineLearning

1. Visualizing & Preprocessing

- engine_location 변수 범주가 1개 밖에 존재하지 않아 삭제 처리

- NA 존재하는 column 및 row 확인(column: "num.of.doors", "bore", "stroke") (row: 18,40,41,42,43)
  -> missForest 패키지 사용하여 random forest 기반 알고리즘 사용하여 NA imputation 처리 진행
 
- 연속형, 범주형 변수 구분하여 list name 저장
  -> 
