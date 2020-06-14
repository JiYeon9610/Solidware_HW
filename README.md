# solidware_MachineLearning

## 1. Visualizing & Preprocessing

 -> engine_location 변수 범주가 1개 밖에 존재하지 않아 삭제 처리

### NA 존재하는 column 및 row 확인
  - (column: "num.of.doors", "bore", "stroke") (row: 18,40,41,42,43)
  - missForest 패키지 사용하여 random forest 기반 알고리즘 사용하여 NA imputation 처리 진행
  
### corrplot 이용하여 연속형 변수들 사이의 correlation 확인
  - 연속형, 범주형 변수 구분하여 list name 저장
  - 유사한 경향을 보이는(correlation이 높은) 두 그룹의 변수들 확인
   1. curb.weight / length / width / price / engine.size
   2. city.mpg / highway.mpg
  
  
### 데이터 개수가 164개인데 비해 X 변수가 24개로 지나치게 많다고 생각하여 정보를 유지하는 방향으로 변수를 축소하고자함
 
  1. PCA 차원 축소
  - 앞서 corrplot으로 확인한 두 그룹의 상관성 시각화하여 확인 (ggpairs 이용)
  - 선형성이 있음을 확인하여 PCA 차원 축소 진행
    1. curb.weight / length / width / price / engine.size -> PC1, PC2(2개 변수)
    2. city.mpg / highway.mpg ->PCA2_2(1개 변수)
    
  2. Random Forest Importance Plot
   - mtry 파라미터 튜닝 후 나온 모델의 importance plot 확인
   - 상위권 5개-> make, height, num of doors, drive.wheels,PC1,PC2 
   
  3. 변수 클러스터링(hclustvar)
   - 앞선 결과에서 상위권에 해당된 변수들은 y를 예측하는데 중요한 역할을 수행할 것이라고 가정
   - 나머지 범주+연속형 변수를 이용하여 변수 클러스터링 진행
   - 4개 그룹
   
  4. 그룹별 clustering 진행(gower distance 이용하여 PAM clustering 진행)
   - 범주형 변수와 연속형 변수가 섞여 있으므로 gower distance 이용
   - 실루엣 값을 이용하여 클러스터 개수 선정
   
  ### 최종독립변수: height, make, drive.wheels, num.of.doors, PC1, PC2, cluster_group1, cluster_group2, cluster_group3, cluster_group4
  
 
 
## 2. Modeling & Evaluating
