# 상점 신용카드 매출 예측
- 소상공인 가맹점 신용카드 빅데이터와 AI로 매출 예측 분석
- 2019년 2월 28일까지의 카드 거래 데이터를 이용하여 2019년 3월 1일 ~ 5월 31일까지의 상점별 3개월 총 매출 예측
- 제공된 데이터의 레코드 단위는 '거래'이며, 예측하고자 하는 레코드의 단위는 3개월 간의 상점 매출임.
#### funda_train.csv : 모델 학습용 데이터

- store_id : 상점의 고유 id
- card_id : 사용한 카드의 고유 아이디
- card_company : 비식별화된 카드 회사
- transcated_date : 거래 날짜
- transacted_time : 거래 시간
- installment_term : 할부 개월 수
- region : 상점 지역
- type_of_business : 상점 업종
- amount : 거래액

#### submission.csv:
- store_id: 상점의 고유 id


### 기본 데이터 구조 설계
- 레코드가 수집된 시간 기준으로 3개월 이후의 총 매출을 예측하도록 구조를 설계해야 함
- ex)
 - 상점 ID: 1, 시점: 4, 특징: 시점 1 ~ 3까지의 상점 1의 특징, 라벨: 시점 5 ~ 7까지의 상점 1의 매출 합계
 - 상점 ID: 1, 시점: 5, 특징: 시점 2 ~ 4까지의 상점 1의 특징, 라벨: 시점 6 ~ 8까지의 상점 1의 매출 합계
 - 상점 ID: 1, 시점: 6, 특징: 시점 3 ~ 5까지의 상점 1의 특징, 라벨: 시점 7 ~ 9까지의 상점 1의 매출 합계
 - 상점 ID: 2136, 시점: 38, 특징: 시점 35 ~ 37까지의 상점 1의 특징, 라벨: 시점 38 ~ 40까지의 상점 1의 매출 합계
 - 상점 ID: 2136, 시점: 38, 특징: 시점 36 ~ 38까지의 상점 1의 특징, 라벨: 시점 39 ~ 41까지의 상점 1의 매출 합계




- 시점의 정의 = ((년-2016)*12 + 월)

## 기초 탐색 및 데이터 준비

#### 학습 데이터 불러오기


```python
import pandas as pd
import os
# df의 시간 범위: 2016-06-01 ~ 2019-02-28
df = pd.read_csv("funda_train.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>card_id</th>
      <th>card_company</th>
      <th>transacted_date</th>
      <th>transacted_time</th>
      <th>installment_term</th>
      <th>region</th>
      <th>type_of_business</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>b</td>
      <td>2016-06-01</td>
      <td>13:13</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>1857.142857</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>h</td>
      <td>2016-06-01</td>
      <td>18:12</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>857.142857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>c</td>
      <td>2016-06-01</td>
      <td>18:52</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>a</td>
      <td>2016-06-01</td>
      <td>20:22</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>7857.142857</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>c</td>
      <td>2016-06-02</td>
      <td>11:06</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>2000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# submission_df의 범위: 2019-03-01 ~ 2019-05-31
submission_df = pd.read_csv("submission.csv")
submission_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### 변수 목록 탐색


```python
# 상점 ID는 일치
print(df['store_id'].unique())
print(submission_df['store_id'].unique())
```

    [   0    1    2 ... 2134 2135 2136]
    [   0    1    2 ... 2134 2135 2136]
    


```python
print(df.columns)
print(submission_df.columns)

# 일치하는 컬럼이 store_id와 amount 밖에 없으므로, 새로운 특징을 추출하는 것이 바람직함
```

    Index(['store_id', 'card_id', 'card_company', 'transacted_date',
           'transacted_time', 'installment_term', 'region', 'type_of_business',
           'amount'],
          dtype='object')
    Index(['store_id', 'amount'], dtype='object')
    

- 예측 대상은 3개월 합계이고, 가지고 있는 데이터는 분단위로 정리되어 있음
- t-2, t-1, t월의 데이터로 t + 1, t + 2, t + 3월의 매출 합계를 예측하는 것으로 문제를 정의
- 따라서 거래 내역을 요약하여 월별로 데이터를 새로 정의하는 것이 중요

## 학습 데이터 구축

#### 년/월 추출

- 기존 시간 변수 transacted_date 에서 연도와 월을 추출


```python
# .str.split을 이용한 년/월 추출
df['transacted_year'] = df['transacted_date'].str.split('-', expand = True).iloc[:, 0].astype(int)
df['transacted_month'] = df['transacted_date'].str.split('-', expand = True).iloc[:, 1].astype(int)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>card_id</th>
      <th>card_company</th>
      <th>transacted_date</th>
      <th>transacted_time</th>
      <th>installment_term</th>
      <th>region</th>
      <th>type_of_business</th>
      <th>amount</th>
      <th>transacted_year</th>
      <th>transacted_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>b</td>
      <td>2016-06-01</td>
      <td>13:13</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>1857.142857</td>
      <td>2016</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>h</td>
      <td>2016-06-01</td>
      <td>18:12</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>857.142857</td>
      <td>2016</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>c</td>
      <td>2016-06-01</td>
      <td>18:52</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>2000.000000</td>
      <td>2016</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>a</td>
      <td>2016-06-01</td>
      <td>20:22</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>7857.142857</td>
      <td>2016</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>c</td>
      <td>2016-06-02</td>
      <td>11:06</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>2000.000000</td>
      <td>2016</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



#### 시점 변수 생성
- 시점 (t) = (연도-2016)*12 + 월


```python
# 데이터 병합을 위한 새로운 컬럼 생성 및 기존 시간 변수 삭제
df['t'] = (df['transacted_year'] - 2016) * 12 + df['transacted_month']
df.drop(['transacted_year', 'transacted_month', 'transacted_date', 'transacted_time'], axis = 1, inplace = True)
```

#### 불필요한 변수 제거
- card_id, card_company는 특징으로 사용하기에는 너무 세분화될 수 있을 뿐만 아니라, 특징으로 유효할 가능성이 없다고 판단하여 삭제


```python
df.drop(['card_id', 'card_company'], axis = 1, inplace = True)
```

#### 업종 특성, 지역, 할부 평균 탐색

- 상태 공간이 매우 큰 범주 변수임을 확인하여, 더미화하기에는 부적절하다고 판단
- 업종 및 지역에 따른 상점 매출 합계의 평균을 사용하기로 결정
- 할부 값은 할부 거래인지 여부만 나타내도록 이진화
- 이 과정에서 결측은 제거하지 않고 없음이라고 변환


```python
df['installment_term'].value_counts().head() # 대부분이 일시불이므로, installment_term 변수를 할부인지 아닌지를 여부로 변환
```




    0    6327632
    3     134709
    2      42101
    5      23751
    6      10792
    Name: installment_term, dtype: int64




```python
df['installment_term'] = (df['installment_term'] > 0).astype(int)
df['installment_term'].value_counts()
```




    0    6327632
    1     228981
    Name: installment_term, dtype: int64




```python
# 상점별 평균 할부 비율
installment_term_per_store = df.groupby(['store_id'])['installment_term'].mean()
installment_term_per_store.head()
```




    store_id
    0    0.038384
    1    0.000000
    2    0.083904
    4    0.001201
    5    0.075077
    Name: installment_term, dtype: float64




```python
# groupby에 결측을 포함시키기 위해, 결측을 문자로 대체
# 지역은 너무 많아서 그대로 활용하기 어려움. 따라서 그대로 더미화하지 않고, 이를 기반으로 한 새로운 변수를 파생해서 사용
df['region'].fillna('없음', inplace = True)
df['region'].value_counts().head()
```




    없음        2042766
    경기 수원시     122029
    충북 청주시     116766
    경남 창원시     107147
    경남 김해시     100673
    Name: region, dtype: int64




```python
df['type_of_business'].value_counts().head()
# 업종도 그 수가 너무 많아 그대로 활용하기 어려움
```




    한식 음식점업    745905
    두발 미용업     178475
    의복 소매업     158234
    기타 주점업     102413
    치킨 전문점      89277
    Name: type_of_business, dtype: int64




```python
# groupby에 결측을 포함시키기 위해, 결측을 문자로 대체
df['type_of_business'].fillna('없음', inplace = True)
```

#### 학습 데이터 구조 작성

- 기존에 정리되지 않은 데이터를 바탕으로 학습 데이터를 생성해야 하는 경우에는 레코드의 단위를 고려하여 학습 데이터의 구조를 먼저 작성하는 것이 바람직함
- funda_train.csv(이하 train_df)에서 store_id, region, type_of_business, t를 기준으로 중복을 제거한 뒤, 해당 컬럼만 갖는 데이터프레임으로 학습 데이터(train_df)를 초기화함


```python
# 'store_id', 'region', 'type_of_business', 't'를 기준으로 중복을 제거한 뒤, 해당 컬럼만 가져옴
train_df = df.drop_duplicates(subset = ['store_id', 'region', 'type_of_business', 't'])[['store_id', 'region', 'type_of_business', 't']]
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>region</th>
      <th>type_of_business</th>
      <th>t</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>6</td>
    </tr>
    <tr>
      <th>145</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>7</td>
    </tr>
    <tr>
      <th>323</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>8</td>
    </tr>
    <tr>
      <th>494</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>9</td>
    </tr>
    <tr>
      <th>654</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



#### 평균 할부율 부착

1. installment_term_per_store 생성
 - store_id에 따른 installment_term의 평균을 groupby를 이용하여 생성 : installment_term_per_store
2. installment_term_per_store를 사전화 : installment_term_per_store.to_dict()

3. train_df의 store_id를 replace 하는 방식으로 평균 할부율 변수 생성



```python
train_df['평균할부율'] = train_df['store_id'].replace(installment_term_per_store.to_dict())
```

#### t-1, t-2, t-3 시점의 매출 합계 부착

- 한 데이터에서는 시점 t를, 다른 데이터에서는 시점 t-1을 붙여야 하는 경우 
 - case 1. t가 유니크한 경우, 각 데이터를 정렬 후, 한 데이터에 대해 shift를 사용한 뒤 concat 수행
 - case 2. t가 유니크하지 않은 경우, t+1 변수를 생성하여 merge 수행


```python
# store_id와 t에 따른 amount 합계 계산: amount_sum_per_t_and_sid
amount_sum_per_t_and_sid = df.groupby(['store_id', 't'], as_index = False)['amount'].sum()
amount_sum_per_t_and_sid.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>t</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>7.470000e+05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>7</td>
      <td>1.005000e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>8</td>
      <td>8.715714e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>9</td>
      <td>8.978571e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>10</td>
      <td>8.354286e+05</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 몇몇 상점은 중간이 비어있음을 확인 => merge에서 문제가 생길 수 있음
amount_sum_per_t_and_sid.groupby(['store_id'])['t'].count().head(10)
```




    store_id
    0     33
    1     33
    2     33
    4     33
    5     33
    6     31
    7     31
    8     28
    9     29
    10    23
    Name: t, dtype: int64




```python
# 따라서 모든 값을 채우기 위해, 피벗 테이블을 생성하고 결측을 바로 앞 값으로 채움
amount_sum_per_t_and_sid = pd.pivot_table(df, values = 'amount', index = 'store_id', columns = 't', aggfunc = 'sum')
amount_sum_per_t_and_sid.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>t</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>...</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
    </tr>
    <tr>
      <th>store_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>747000.000000</td>
      <td>1.005000e+06</td>
      <td>871571.428571</td>
      <td>8.978571e+05</td>
      <td>8.354286e+05</td>
      <td>6.970000e+05</td>
      <td>7.618571e+05</td>
      <td>585642.857143</td>
      <td>7.940000e+05</td>
      <td>720257.142857</td>
      <td>...</td>
      <td>6.864286e+05</td>
      <td>7.072857e+05</td>
      <td>7.587143e+05</td>
      <td>6.798571e+05</td>
      <td>6.518571e+05</td>
      <td>7.390000e+05</td>
      <td>6.760000e+05</td>
      <td>8.745714e+05</td>
      <td>6.828571e+05</td>
      <td>5.152857e+05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>137214.285714</td>
      <td>1.630000e+05</td>
      <td>118142.857143</td>
      <td>9.042857e+04</td>
      <td>1.180714e+05</td>
      <td>1.118571e+05</td>
      <td>1.155714e+05</td>
      <td>129642.857143</td>
      <td>1.602143e+05</td>
      <td>168428.571429</td>
      <td>...</td>
      <td>8.050000e+04</td>
      <td>7.828571e+04</td>
      <td>1.007857e+05</td>
      <td>9.214286e+04</td>
      <td>6.357143e+04</td>
      <td>9.500000e+04</td>
      <td>8.078571e+04</td>
      <td>8.528571e+04</td>
      <td>1.482857e+05</td>
      <td>7.742857e+04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>260714.285714</td>
      <td>8.285714e+04</td>
      <td>131428.571429</td>
      <td>1.428571e+05</td>
      <td>1.097143e+05</td>
      <td>1.985714e+05</td>
      <td>1.600000e+05</td>
      <td>180714.285714</td>
      <td>1.542857e+05</td>
      <td>43571.428571</td>
      <td>...</td>
      <td>4.728571e+05</td>
      <td>3.542857e+05</td>
      <td>6.892857e+05</td>
      <td>4.578571e+05</td>
      <td>4.807143e+05</td>
      <td>5.100000e+05</td>
      <td>1.854286e+05</td>
      <td>3.407143e+05</td>
      <td>4.078571e+05</td>
      <td>4.968571e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>733428.571429</td>
      <td>7.689286e+05</td>
      <td>698428.571429</td>
      <td>9.364286e+05</td>
      <td>7.627143e+05</td>
      <td>8.595714e+05</td>
      <td>1.069857e+06</td>
      <td>689142.857143</td>
      <td>1.050143e+06</td>
      <td>970285.714286</td>
      <td>...</td>
      <td>7.754286e+05</td>
      <td>8.812857e+05</td>
      <td>1.050929e+06</td>
      <td>8.492857e+05</td>
      <td>6.981429e+05</td>
      <td>8.284286e+05</td>
      <td>8.830000e+05</td>
      <td>9.238571e+05</td>
      <td>9.448571e+05</td>
      <td>8.822857e+05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>342500.000000</td>
      <td>4.327143e+05</td>
      <td>263500.000000</td>
      <td>2.321429e+05</td>
      <td>2.115714e+05</td>
      <td>1.820857e+05</td>
      <td>1.475714e+05</td>
      <td>120957.142857</td>
      <td>1.864286e+05</td>
      <td>169000.000000</td>
      <td>...</td>
      <td>4.438571e+05</td>
      <td>5.637143e+05</td>
      <td>6.070714e+05</td>
      <td>4.828857e+05</td>
      <td>1.950000e+05</td>
      <td>3.249286e+05</td>
      <td>3.833000e+05</td>
      <td>3.995714e+05</td>
      <td>3.230000e+05</td>
      <td>2.155143e+05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>568857.142857</td>
      <td>1.440143e+06</td>
      <td>1.238857e+06</td>
      <td>1.055429e+06</td>
      <td>9.268571e+05</td>
      <td>885642.857143</td>
      <td>8.003571e+05</td>
      <td>930714.285714</td>
      <td>...</td>
      <td>1.808357e+06</td>
      <td>1.752286e+06</td>
      <td>1.583786e+06</td>
      <td>1.628786e+06</td>
      <td>2.074071e+06</td>
      <td>1.907643e+06</td>
      <td>2.389143e+06</td>
      <td>2.230286e+06</td>
      <td>2.015500e+06</td>
      <td>2.463857e+06</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>107857.142857</td>
      <td>3.756429e+05</td>
      <td>3.236429e+05</td>
      <td>3.450000e+05</td>
      <td>2.914286e+05</td>
      <td>231614.285714</td>
      <td>2.713571e+05</td>
      <td>249857.142857</td>
      <td>...</td>
      <td>2.657143e+05</td>
      <td>4.195429e+05</td>
      <td>4.628429e+05</td>
      <td>4.231286e+05</td>
      <td>3.203286e+05</td>
      <td>4.200286e+05</td>
      <td>3.143857e+05</td>
      <td>3.024143e+05</td>
      <td>1.364714e+05</td>
      <td>5.797143e+04</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.925714e+05</td>
      <td>7.355000e+05</td>
      <td>467857.142857</td>
      <td>4.756429e+05</td>
      <td>603500.000000</td>
      <td>...</td>
      <td>1.837429e+06</td>
      <td>1.359857e+06</td>
      <td>1.213543e+06</td>
      <td>1.086000e+06</td>
      <td>1.369557e+06</td>
      <td>1.272071e+06</td>
      <td>1.260557e+06</td>
      <td>1.157257e+06</td>
      <td>1.134671e+06</td>
      <td>1.298329e+06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.071429e+05</td>
      <td>6.371429e+05</td>
      <td>6.035714e+05</td>
      <td>225428.571429</td>
      <td>2.871429e+05</td>
      <td>344428.571429</td>
      <td>...</td>
      <td>6.385714e+05</td>
      <td>2.765714e+05</td>
      <td>3.400000e+05</td>
      <td>2.542857e+05</td>
      <td>9.265714e+05</td>
      <td>8.714286e+05</td>
      <td>6.928571e+05</td>
      <td>6.628571e+05</td>
      <td>3.700000e+05</td>
      <td>4.057143e+05</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2.902857e+05</td>
      <td>6.078571e+05</td>
      <td>4.445714e+05</td>
      <td>6.414286e+05</td>
      <td>7.955714e+05</td>
      <td>4.992857e+05</td>
      <td>5.901429e+05</td>
      <td>5.184286e+05</td>
      <td>5.251429e+05</td>
      <td>6.548571e+05</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 33 columns</p>
</div>




```python
# 따라서 모든 값을 채우기 위해, 피벗 테이블을 생성하고 결측을 바로 앞 값으로 채움
amount_sum_per_t_and_sid = amount_sum_per_t_and_sid.fillna(method = 'ffill', axis = 1).fillna(method = 'bfill', axis = 1)
amount_sum_per_t_and_sid.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>t</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>...</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
    </tr>
    <tr>
      <th>store_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>747000.000000</td>
      <td>1.005000e+06</td>
      <td>871571.428571</td>
      <td>8.978571e+05</td>
      <td>8.354286e+05</td>
      <td>6.970000e+05</td>
      <td>7.618571e+05</td>
      <td>585642.857143</td>
      <td>7.940000e+05</td>
      <td>720257.142857</td>
      <td>...</td>
      <td>6.864286e+05</td>
      <td>7.072857e+05</td>
      <td>7.587143e+05</td>
      <td>6.798571e+05</td>
      <td>6.518571e+05</td>
      <td>7.390000e+05</td>
      <td>6.760000e+05</td>
      <td>8.745714e+05</td>
      <td>6.828571e+05</td>
      <td>5.152857e+05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>137214.285714</td>
      <td>1.630000e+05</td>
      <td>118142.857143</td>
      <td>9.042857e+04</td>
      <td>1.180714e+05</td>
      <td>1.118571e+05</td>
      <td>1.155714e+05</td>
      <td>129642.857143</td>
      <td>1.602143e+05</td>
      <td>168428.571429</td>
      <td>...</td>
      <td>8.050000e+04</td>
      <td>7.828571e+04</td>
      <td>1.007857e+05</td>
      <td>9.214286e+04</td>
      <td>6.357143e+04</td>
      <td>9.500000e+04</td>
      <td>8.078571e+04</td>
      <td>8.528571e+04</td>
      <td>1.482857e+05</td>
      <td>7.742857e+04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>260714.285714</td>
      <td>8.285714e+04</td>
      <td>131428.571429</td>
      <td>1.428571e+05</td>
      <td>1.097143e+05</td>
      <td>1.985714e+05</td>
      <td>1.600000e+05</td>
      <td>180714.285714</td>
      <td>1.542857e+05</td>
      <td>43571.428571</td>
      <td>...</td>
      <td>4.728571e+05</td>
      <td>3.542857e+05</td>
      <td>6.892857e+05</td>
      <td>4.578571e+05</td>
      <td>4.807143e+05</td>
      <td>5.100000e+05</td>
      <td>1.854286e+05</td>
      <td>3.407143e+05</td>
      <td>4.078571e+05</td>
      <td>4.968571e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>733428.571429</td>
      <td>7.689286e+05</td>
      <td>698428.571429</td>
      <td>9.364286e+05</td>
      <td>7.627143e+05</td>
      <td>8.595714e+05</td>
      <td>1.069857e+06</td>
      <td>689142.857143</td>
      <td>1.050143e+06</td>
      <td>970285.714286</td>
      <td>...</td>
      <td>7.754286e+05</td>
      <td>8.812857e+05</td>
      <td>1.050929e+06</td>
      <td>8.492857e+05</td>
      <td>6.981429e+05</td>
      <td>8.284286e+05</td>
      <td>8.830000e+05</td>
      <td>9.238571e+05</td>
      <td>9.448571e+05</td>
      <td>8.822857e+05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>342500.000000</td>
      <td>4.327143e+05</td>
      <td>263500.000000</td>
      <td>2.321429e+05</td>
      <td>2.115714e+05</td>
      <td>1.820857e+05</td>
      <td>1.475714e+05</td>
      <td>120957.142857</td>
      <td>1.864286e+05</td>
      <td>169000.000000</td>
      <td>...</td>
      <td>4.438571e+05</td>
      <td>5.637143e+05</td>
      <td>6.070714e+05</td>
      <td>4.828857e+05</td>
      <td>1.950000e+05</td>
      <td>3.249286e+05</td>
      <td>3.833000e+05</td>
      <td>3.995714e+05</td>
      <td>3.230000e+05</td>
      <td>2.155143e+05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>568857.142857</td>
      <td>5.688571e+05</td>
      <td>568857.142857</td>
      <td>1.440143e+06</td>
      <td>1.238857e+06</td>
      <td>1.055429e+06</td>
      <td>9.268571e+05</td>
      <td>885642.857143</td>
      <td>8.003571e+05</td>
      <td>930714.285714</td>
      <td>...</td>
      <td>1.808357e+06</td>
      <td>1.752286e+06</td>
      <td>1.583786e+06</td>
      <td>1.628786e+06</td>
      <td>2.074071e+06</td>
      <td>1.907643e+06</td>
      <td>2.389143e+06</td>
      <td>2.230286e+06</td>
      <td>2.015500e+06</td>
      <td>2.463857e+06</td>
    </tr>
    <tr>
      <th>7</th>
      <td>107857.142857</td>
      <td>1.078571e+05</td>
      <td>107857.142857</td>
      <td>3.756429e+05</td>
      <td>3.236429e+05</td>
      <td>3.450000e+05</td>
      <td>2.914286e+05</td>
      <td>231614.285714</td>
      <td>2.713571e+05</td>
      <td>249857.142857</td>
      <td>...</td>
      <td>2.657143e+05</td>
      <td>4.195429e+05</td>
      <td>4.628429e+05</td>
      <td>4.231286e+05</td>
      <td>3.203286e+05</td>
      <td>4.200286e+05</td>
      <td>3.143857e+05</td>
      <td>3.024143e+05</td>
      <td>1.364714e+05</td>
      <td>5.797143e+04</td>
    </tr>
    <tr>
      <th>8</th>
      <td>192571.428571</td>
      <td>1.925714e+05</td>
      <td>192571.428571</td>
      <td>1.925714e+05</td>
      <td>1.925714e+05</td>
      <td>1.925714e+05</td>
      <td>7.355000e+05</td>
      <td>467857.142857</td>
      <td>4.756429e+05</td>
      <td>603500.000000</td>
      <td>...</td>
      <td>1.837429e+06</td>
      <td>1.359857e+06</td>
      <td>1.213543e+06</td>
      <td>1.086000e+06</td>
      <td>1.369557e+06</td>
      <td>1.272071e+06</td>
      <td>1.260557e+06</td>
      <td>1.157257e+06</td>
      <td>1.134671e+06</td>
      <td>1.298329e+06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>107142.857143</td>
      <td>1.071429e+05</td>
      <td>107142.857143</td>
      <td>1.071429e+05</td>
      <td>1.071429e+05</td>
      <td>6.371429e+05</td>
      <td>6.035714e+05</td>
      <td>225428.571429</td>
      <td>2.871429e+05</td>
      <td>344428.571429</td>
      <td>...</td>
      <td>6.385714e+05</td>
      <td>2.765714e+05</td>
      <td>3.400000e+05</td>
      <td>2.542857e+05</td>
      <td>9.265714e+05</td>
      <td>8.714286e+05</td>
      <td>6.928571e+05</td>
      <td>6.628571e+05</td>
      <td>3.700000e+05</td>
      <td>4.057143e+05</td>
    </tr>
    <tr>
      <th>10</th>
      <td>496714.285714</td>
      <td>4.967143e+05</td>
      <td>496714.285714</td>
      <td>4.967143e+05</td>
      <td>4.967143e+05</td>
      <td>4.967143e+05</td>
      <td>4.967143e+05</td>
      <td>496714.285714</td>
      <td>4.967143e+05</td>
      <td>496714.285714</td>
      <td>...</td>
      <td>2.902857e+05</td>
      <td>6.078571e+05</td>
      <td>4.445714e+05</td>
      <td>6.414286e+05</td>
      <td>7.955714e+05</td>
      <td>4.992857e+05</td>
      <td>5.901429e+05</td>
      <td>5.184286e+05</td>
      <td>5.251429e+05</td>
      <td>6.548571e+05</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 33 columns</p>
</div>




```python
# stack을 이용하여, 컬럼도 행 인덱스로 밀어넣고, 인덱스를 초기화하여 인덱스를 컬럼으로 가져옴
amount_sum_per_t_and_sid = amount_sum_per_t_and_sid.stack().reset_index()
amount_sum_per_t_and_sid.rename({0:"amount"}, axis = 1, inplace = True)
```


```python
# t - k (k = 1, 2, 3) 시점의 부착
# train_df의 t는 amount_sum_per_t_and_sid의 t-k과 부착되어야 하므로, amount_sum_per_t_and_sid의 t에 k를 더함

for k in range(1, 4):
    amount_sum_per_t_and_sid['t_{}'.format(k)] = amount_sum_per_t_and_sid['t'] + k
    train_df = pd.merge(train_df, amount_sum_per_t_and_sid.drop('t', axis = 1), left_on = ['store_id', 't'], right_on = ['store_id', 't_{}'.format(k)])
    
    # 부착한 뒤, 불필요한 변수 제거 및 변수명 변경: 다음 이터레이션에서의 병합이 잘되게 하기 위해서
    train_df.rename({"amount":"{}_before_amount".format(k)}, axis = 1, inplace = True)
    train_df.drop(['t_{}'.format(k)], axis = 1, inplace = True)
    amount_sum_per_t_and_sid.drop(['t_{}'.format(k)], axis = 1, inplace = True)
    
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>region</th>
      <th>type_of_business</th>
      <th>t</th>
      <th>평균할부율</th>
      <th>1_before_amount</th>
      <th>2_before_amount</th>
      <th>3_before_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>9</td>
      <td>0.038384</td>
      <td>871571.428571</td>
      <td>1.005000e+06</td>
      <td>7.470000e+05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>10</td>
      <td>0.038384</td>
      <td>897857.142857</td>
      <td>8.715714e+05</td>
      <td>1.005000e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>11</td>
      <td>0.038384</td>
      <td>835428.571429</td>
      <td>8.978571e+05</td>
      <td>8.715714e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>12</td>
      <td>0.038384</td>
      <td>697000.000000</td>
      <td>8.354286e+05</td>
      <td>8.978571e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>13</td>
      <td>0.038384</td>
      <td>761857.142857</td>
      <td>6.970000e+05</td>
      <td>8.354286e+05</td>
    </tr>
  </tbody>
</table>
</div>



#### t-1, t-2, t-3 시점의 지역별 매출 합계 평균 부착

### 기존 매출 합계 부착
- store_id 와 t에 따른 amount의 합계 계산 : amount_sum_per_t_and_sid

- 다음 과정을 k = 1, 2, 3에 대해 반복
1. amount_sum_per_t_and_sid 에 t_k 변수 생성 (t_k = t + k)
2. train_df와 amount_sum_per_t_and_sid 병합 (단, amount_sum_per_t_and_sid 에는 t 컬럼 삭제 )
3. 병합 후 train_df의 amount 변수명을 k_before_amount로 변경
4. 불필요한 변수가 추가되는 것을 막기 위해, amount_sum_per_t_and_sid와 train_df에 t_k 변수 삭제

### 기존 지역별 매출 합계 부착
1. store_id를 키로 하고, region을 value로 하는 사전 생성
2. amount_sum_per_t_and_sid에서 region 변수 생성 및 region과 t에 따른 amount 평균 계산: amount_mean_per_t_and_region
3. 다음 과정을 k = 1, 2, 3 에 대해 반복
 1. amount_mean_per_t_and_region에 t_k변수 생성 (t_k = t + k)
 2. train_df와 amount_mean_per_t_and_region 병합 (단, amount_mean_per_t_and_region에는 t컬럼 삭제)
 3. 병합 후 train_df의 amount 변수명을 k_before_amount_of_region로 변경
 4. 불필요한 변수가 추가되는 것을 막기 위해, amount_sum_per_t_and_sid와 train_df에 t_k 변수 삭제


```python
# amount_sum_per_t_and_sid의 store_id를 region으로 대체시키기
store_to_region = df[['store_id', 'region']].drop_duplicates().set_index(['store_id'])['region'].to_dict()
amount_sum_per_t_and_sid['region'] = amount_sum_per_t_and_sid['store_id'].replace(store_to_region)

# 지역별 평균 매출 계산
amount_mean_per_t_and_region = amount_sum_per_t_and_sid.groupby(['region', 't'], as_index = False)['amount'].mean()
```


```python
# t - k (k = 1, 2, 3) 시점의 부착

for k in range(1, 4):
    amount_mean_per_t_and_region['t_{}'.format(k)] = amount_mean_per_t_and_region['t'] + k
    train_df = pd.merge(train_df, amount_mean_per_t_and_region.drop('t', axis = 1), left_on = ['region', 't'], right_on = ['region', 't_{}'.format(k)])
    train_df.rename({"amount":"{}_before_amount_of_region".format(k)}, axis = 1, inplace = True)
    
    train_df.drop(['t_{}'.format(k)], axis = 1, inplace = True)
    amount_mean_per_t_and_region.drop(['t_{}'.format(k)], axis = 1, inplace = True)    
    
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>region</th>
      <th>type_of_business</th>
      <th>t</th>
      <th>평균할부율</th>
      <th>1_before_amount</th>
      <th>2_before_amount</th>
      <th>3_before_amount</th>
      <th>1_before_amount_of_region</th>
      <th>2_before_amount_of_region</th>
      <th>3_before_amount_of_region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>9</td>
      <td>0.038384</td>
      <td>871571.428571</td>
      <td>1.005000e+06</td>
      <td>747000.000000</td>
      <td>761987.421532</td>
      <td>756108.674948</td>
      <td>739654.068323</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>없음</td>
      <td>없음</td>
      <td>9</td>
      <td>0.000000</td>
      <td>118142.857143</td>
      <td>1.630000e+05</td>
      <td>137214.285714</td>
      <td>761987.421532</td>
      <td>756108.674948</td>
      <td>739654.068323</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>없음</td>
      <td>없음</td>
      <td>9</td>
      <td>0.083904</td>
      <td>131428.571429</td>
      <td>8.285714e+04</td>
      <td>260714.285714</td>
      <td>761987.421532</td>
      <td>756108.674948</td>
      <td>739654.068323</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>없음</td>
      <td>의복 액세서리 및 모조 장신구 도매업</td>
      <td>9</td>
      <td>0.075077</td>
      <td>263500.000000</td>
      <td>4.327143e+05</td>
      <td>342500.000000</td>
      <td>761987.421532</td>
      <td>756108.674948</td>
      <td>739654.068323</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>없음</td>
      <td>없음</td>
      <td>9</td>
      <td>0.011558</td>
      <td>107857.142857</td>
      <td>1.078571e+05</td>
      <td>107857.142857</td>
      <td>761987.421532</td>
      <td>756108.674948</td>
      <td>739654.068323</td>
    </tr>
  </tbody>
</table>
</div>



#### t-1, t-2, t-3 시점의 업종별 매출 합계 평균 부착


### 기존 업종별 매출 합계 부착
1. store_id를 키로 설정 , type_of_business를 value로 하는 사전 생성
2. amount_sum_per_t_and_sid에서 type_of_business 변수 생성 및 type_of_business와 t에 따른 amount 평균 계산 : amount_mean_per_t_and_type_of_business
3. 다음 과정을 k = 1, 2, 3 에 대해 반복
 1. amount_mean_per_t_and_ type_of_business에 t_k 변수 생성 (t_k = t + k)
 2. train_df와 amount_mean_per_t_and_ type_of_business 병합 ( 단, type_of_business에는 t 컬럼 삭제 )
 3. 병합 후 train_df의 amount 변수명을 k_before_amount_of_region로 변경
 4. 불필요한 변수가 추가되는 것을 막기 위해, type_of_business와 train_df에 t_k 변수 삭제
 
### 라벨 부착하기
1. 다음 과정을 k = 1, 2, 3 에 대해 반복
 1. amount_sum_per_t_and_sid에 t_k (t_k = t – k) 변수 생성
 2. train_df와 amount_sum_per_t_and_sid를 병합
 3. 병합 후, train_df의 amount 변수명을 Y_k 로 변경
2. 라벨 생성 : Y = Y_1 + Y_2 + Y_3


```python
# amount_sum_per_t_and_sid의 store_id를 type_of_business으로 대체시키기
store_to_type_of_business = df[['store_id', 'type_of_business']].drop_duplicates().set_index(['store_id'])['type_of_business'].to_dict()
amount_sum_per_t_and_sid['type_of_business'] = amount_sum_per_t_and_sid['store_id'].replace(store_to_type_of_business)

# 지역별 평균 매출 계산
amount_mean_per_t_and_type_of_business = amount_sum_per_t_and_sid.groupby(['type_of_business', 't'], as_index = False)['amount'].mean()
```


```python
# t - k (k = 1, 2, 3) 시점의 부착
# train_df의 t는 amount_sum_per_t_and_sid의 t-k과 부착되어야 하므로, amount_sum_per_t_and_sid의 t에 k를 더함

for k in range(1, 4):
    amount_mean_per_t_and_type_of_business['t_{}'.format(k)] = amount_mean_per_t_and_type_of_business['t'] + k
    train_df = pd.merge(train_df, amount_mean_per_t_and_type_of_business.drop('t', axis = 1), left_on = ['type_of_business', 't'], right_on = ['type_of_business', 't_{}'.format(k)])
    train_df.rename({"amount":"{}_before_amount_of_type_of_business".format(k)}, axis = 1, inplace = True)
    
    train_df.drop(['t_{}'.format(k)], axis = 1, inplace = True)
    amount_mean_per_t_and_type_of_business.drop(['t_{}'.format(k)], axis = 1, inplace = True)       
    
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>region</th>
      <th>type_of_business</th>
      <th>t</th>
      <th>평균할부율</th>
      <th>1_before_amount</th>
      <th>2_before_amount</th>
      <th>3_before_amount</th>
      <th>1_before_amount_of_region</th>
      <th>2_before_amount_of_region</th>
      <th>3_before_amount_of_region</th>
      <th>1_before_amount_of_type_of_business</th>
      <th>2_before_amount_of_type_of_business</th>
      <th>3_before_amount_of_type_of_business</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>9</td>
      <td>0.038384</td>
      <td>871571.428571</td>
      <td>1.005000e+06</td>
      <td>747000.000000</td>
      <td>7.619874e+05</td>
      <td>7.561087e+05</td>
      <td>7.396541e+05</td>
      <td>761025.0</td>
      <td>804979.761905</td>
      <td>679950.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>792</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>9</td>
      <td>0.218887</td>
      <td>681142.857143</td>
      <td>8.808571e+05</td>
      <td>733714.285714</td>
      <td>7.619874e+05</td>
      <td>7.561087e+05</td>
      <td>7.396541e+05</td>
      <td>761025.0</td>
      <td>804979.761905</td>
      <td>679950.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23</td>
      <td>경기 안양시</td>
      <td>기타 미용업</td>
      <td>9</td>
      <td>0.048795</td>
      <td>879242.857143</td>
      <td>7.308571e+05</td>
      <td>845285.714286</td>
      <td>8.288317e+05</td>
      <td>5.887330e+05</td>
      <td>9.559733e+05</td>
      <td>761025.0</td>
      <td>804979.761905</td>
      <td>679950.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>192</td>
      <td>경기 화성시</td>
      <td>기타 미용업</td>
      <td>9</td>
      <td>0.100542</td>
      <td>579000.000000</td>
      <td>5.234286e+05</td>
      <td>551142.857143</td>
      <td>1.234460e+06</td>
      <td>1.227921e+06</td>
      <td>1.180455e+06</td>
      <td>761025.0</td>
      <td>804979.761905</td>
      <td>679950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536</td>
      <td>서울 광진구</td>
      <td>기타 미용업</td>
      <td>9</td>
      <td>0.014810</td>
      <td>96285.714286</td>
      <td>7.985714e+04</td>
      <td>99857.142857</td>
      <td>3.786820e+06</td>
      <td>3.397973e+06</td>
      <td>3.524075e+06</td>
      <td>761025.0</td>
      <td>804979.761905</td>
      <td>679950.0</td>
    </tr>
  </tbody>
</table>
</div>



#### 라벨 부착하기


```python
# 현 시점에서 t + 1, t + 2, t + 3의 매출을 부착해야 함
```


```python
amount_sum_per_t_and_sid.drop(['region', 'type_of_business'], axis = 1, inplace = True)
for k in range(1, 4):
    amount_sum_per_t_and_sid['t_{}'.format(k)] = amount_sum_per_t_and_sid['t'] - k   
    train_df = pd.merge(train_df, amount_sum_per_t_and_sid.drop('t', axis = 1), left_on = ['store_id', 't'], right_on = ['store_id', 't_{}'.format(k)])
    train_df.rename({"amount": "Y_{}".format(k)}, axis = 1, inplace = True)
    
    train_df.drop(['t_{}'.format(k)], axis = 1, inplace = True)
    amount_sum_per_t_and_sid.drop(['t_{}'.format(k)], axis = 1, inplace = True)      
```


```python
train_df['Y'] = train_df['Y_1'] + train_df['Y_2'] + train_df['Y_3']
```

## 학습 데이터 탐색 및 전처리

#### 특징과 라벨 분리


```python
X = train_df.drop(['store_id', 'region', 'type_of_business', 't', 'Y_1', 'Y_2', 'Y_3', 'Y'], axis = 1)
Y = train_df['Y']
```

#### 데이터 분할 및 구조 탐색


```python
from sklearn.model_selection import train_test_split
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y)
Train_X.shape # 특징 대비 샘플이 많음
```




    (37673, 10)




```python
Train_Y.describe()
```




    count    3.767300e+04
    mean     3.455220e+06
    std      5.392722e+06
    min      0.000000e+00
    25%      1.131571e+06
    50%      2.215564e+06
    75%      4.100429e+06
    max      1.727659e+08
    Name: Y, dtype: float64



#### 이상치 제거


```python
import numpy as np
def IQR_rule(val_list): # 한 특징에 포함된 값 (열 벡터)
    # IQR 계산    
    Q1 = np.quantile(val_list, 0.25)
    Q3 = np.quantile(val_list, 0.75)
    IQR = Q3 - Q1
    
    # IQR rule을 위배하지 않는 bool list 계산 (True: 이상치 X, False: 이상치 O)
    not_outlier_condition = (Q3 + 1.5 * IQR > val_list) & (Q1 - 1.5 * IQR < val_list)
    return not_outlier_condition
```


```python
Y_condition = IQR_rule(Train_Y)
Train_Y = Train_Y[Y_condition]
Train_X = Train_X[Y_condition]
```

#### 치우침 제거


```python
# 모두 좌로 치우침을 확인
Train_X.skew()
```




    평균할부율                                  2.979842
    1_before_amount                        2.444171
    2_before_amount                        2.428243
    3_before_amount                        2.463657
    1_before_amount_of_region              3.244259
    2_before_amount_of_region              3.206229
    3_before_amount_of_region              3.200283
    1_before_amount_of_type_of_business    1.793848
    2_before_amount_of_type_of_business    1.910635
    3_before_amount_of_type_of_business    1.885685
    dtype: float64




```python
# 치우침 제거
import numpy as np
biased_variables = Train_X.columns[Train_X.skew().abs() > 1.5] # 왜도의 절대값이 1.5 이상인 컬럼만 가져오기
Train_X[biased_variables] = Train_X[biased_variables] - Train_X[biased_variables].min() + 1
Train_X[biased_variables] = np.sqrt(Train_X[biased_variables])
```


```python
Train_X.skew()
```




    평균할부율                                  2.769802
    1_before_amount                        0.624789
    2_before_amount                        0.726716
    3_before_amount                        0.737180
    1_before_amount_of_region              1.811923
    2_before_amount_of_region              1.779948
    3_before_amount_of_region              1.774822
    1_before_amount_of_type_of_business   -0.275585
    2_before_amount_of_type_of_business   -0.270576
    3_before_amount_of_type_of_business   -0.247877
    dtype: float64



#### 스케일링 수행


```python
Train_X.max() - Train_X.min() # 특징 간 스케일 차이가 큼을 확인 => 스케일이 작은 특징은 영향을 거의 주지 못할 것이라 예상됨
```




    평균할부율                                     0.379933
    1_before_amount                        3688.250931
    2_before_amount                        3595.029696
    3_before_amount                        3518.070636
    1_before_amount_of_region              2588.236153
    2_before_amount_of_region              2588.236153
    3_before_amount_of_region              2588.236153
    1_before_amount_of_type_of_business    2581.717478
    2_before_amount_of_type_of_business    2614.039115
    3_before_amount_of_type_of_business    2341.952868
    dtype: float64




```python
# 스케일링 수행
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(Train_X)
s_Train_X = scaler.transform(Train_X)
s_Test_X = scaler.transform(Test_X)

Train_X = pd.DataFrame(s_Train_X, columns = Train_X.columns)
Test_X = pd.DataFrame(s_Test_X, columns = Train_X.columns)

del s_Train_X, s_Test_X # 메모리 관리를 위해, 불필요한 값은 제거
```

#### 모델 학습 

샘플 대비 특징이 적고, 특징의 타입이 전부 연속형으로 같음. 따라서 아래 세 개의 모델 및 특징 선택 기준을 고려

- 모델 1. kNN
- 모델 2. RandomForestRegressor
- 모델 3. LightGBM

- 특징 선택: 3 ~ 10개 (기준: f_regression)


```python
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import RandomForestRegressor as RFR
from lightgbm import LGBMRegressor as LGB
from sklearn.feature_selection import *
```


```python
# 파라미터 그리드 생성
param_grid = dict() 
# 입력: 모델 함수, 출력: 모델의 하이퍼 파라미터 그리드

# 모델별 파라미터 그리드 생성
param_grid_for_knn = ParameterGrid({"n_neighbors": [1, 3, 5, 7],
                           "metric":['euclidean', 'cosine']})

param_grid_for_RFR = ParameterGrid({"max_depth": [1, 2, 3, 4],
                           "n_estimators":[100, 200],
                                   "max_samples":[0.5, 0.6, 0.7, None]}) # 특징 대비 샘플이 많아서 붓스트랩 비율 (max_samples)을 설정 

param_grid_for_LGB = ParameterGrid({"max_depth": [1, 2, 3, 4],
                                   "n_estimators":[100, 200],
                            "learning_rate": [0.05, 0.1, 0.15]})

# 모델 - 하이퍼 파라미터 그리드를 param_grid에 추가
param_grid[KNN] = param_grid_for_knn
param_grid[RFR] = param_grid_for_RFR
param_grid[LGB] = param_grid_for_LGB
```


```python
# 출력을 위한 max_iter_num 계산
max_iter_num = 0
for k in range(10, 2, -1):
    for M in param_grid.keys():
        for P in param_grid[M]:
            max_iter_num += 1
           
from sklearn.metrics import mean_absolute_error as MAE

best_score = 9999999999
iteration_num = 0
for k in range(10, 2, -1): # 메모리 부담 해소를 위해, 1씩 감소시킴
    selector = SelectKBest(f_regression, k = k).fit(Train_X, Train_Y)
    selected_features = Train_X.columns[selector.get_support()]

    Train_X = Train_X[selected_features]
    Test_X = Test_X[selected_features]
    
    for M in param_grid.keys():
        for P in param_grid[M]:
            # LightGBM에서 DataFrame이 잘 처리되지 않는 것을 방지하기 위해 .values를 사용
            model = M(**P).fit(Train_X.values, Train_Y.values)
            pred_Y = model.predict(Test_X.values)
            score = MAE(Test_Y.values, pred_Y)
            
            if score < best_score:
                best_score = score
                best_model = M
                best_paramter = P
                best_features = selected_features    
                
            iteration_num += 1
            print("iter_num:{}/{}, score: {}, best_score: {}".format(iteration_num, max_iter_num, round(score, 2), round(best_score, 2)))
```

    iter_num:1/512, score: 4226292.7, best_score: 4226292.7
    iter_num:2/512, score: 3966742.96, best_score: 3966742.96
    iter_num:3/512, score: 4327727.63, best_score: 3966742.96
    iter_num:4/512, score: 4386552.03, best_score: 3966742.96
    iter_num:5/512, score: 1956342.02, best_score: 1956342.02
    iter_num:6/512, score: 1696555.71, best_score: 1696555.71
    iter_num:7/512, score: 1622245.04, best_score: 1622245.04
    iter_num:8/512, score: 1586524.15, best_score: 1586524.15
    iter_num:9/512, score: 3032534.15, best_score: 1586524.15
    iter_num:10/512, score: 3033367.64, best_score: 1586524.15
    iter_num:11/512, score: 3035198.11, best_score: 1586524.15
    iter_num:12/512, score: 3039050.95, best_score: 1586524.15
    iter_num:13/512, score: 3033930.11, best_score: 1586524.15
    iter_num:14/512, score: 3036046.92, best_score: 1586524.15
    iter_num:15/512, score: 3037878.48, best_score: 1586524.15
    iter_num:16/512, score: 3037848.18, best_score: 1586524.15
    iter_num:17/512, score: 3812271.62, best_score: 1586524.15
    iter_num:18/512, score: 3797065.21, best_score: 1586524.15
    iter_num:19/512, score: 3800957.67, best_score: 1586524.15
    iter_num:20/512, score: 3802975.87, best_score: 1586524.15
    iter_num:21/512, score: 3797870.24, best_score: 1586524.15
    iter_num:22/512, score: 3802631.54, best_score: 1586524.15
    iter_num:23/512, score: 3800425.37, best_score: 1586524.15
    iter_num:24/512, score: 3803447.1, best_score: 1586524.15
    iter_num:25/512, score: 4215743.02, best_score: 1586524.15
    iter_num:26/512, score: 4209712.61, best_score: 1586524.15
    iter_num:27/512, score: 4208106.52, best_score: 1586524.15
    iter_num:28/512, score: 4209494.69, best_score: 1586524.15
    iter_num:29/512, score: 4207287.69, best_score: 1586524.15
    iter_num:30/512, score: 4215321.96, best_score: 1586524.15
    iter_num:31/512, score: 4190359.82, best_score: 1586524.15
    iter_num:32/512, score: 4188814.61, best_score: 1586524.15
    iter_num:33/512, score: 4450693.97, best_score: 1586524.15
    iter_num:34/512, score: 4424514.37, best_score: 1586524.15
    iter_num:35/512, score: 4476640.42, best_score: 1586524.15
    iter_num:36/512, score: 4438892.83, best_score: 1586524.15
    iter_num:37/512, score: 4454595.18, best_score: 1586524.15
    iter_num:38/512, score: 4439473.54, best_score: 1586524.15
    iter_num:39/512, score: 4450131.33, best_score: 1586524.15
    iter_num:40/512, score: 4451467.75, best_score: 1586524.15
    iter_num:41/512, score: 4148851.66, best_score: 1586524.15
    iter_num:42/512, score: 4466258.71, best_score: 1586524.15
    iter_num:43/512, score: 4432405.24, best_score: 1586524.15
    iter_num:44/512, score: 4537717.89, best_score: 1586524.15
    iter_num:45/512, score: 4917942.43, best_score: 1586524.15
    iter_num:46/512, score: 5227962.27, best_score: 1586524.15
    iter_num:47/512, score: 4867477.94, best_score: 1586524.15
    iter_num:48/512, score: 5185396.33, best_score: 1586524.15
    iter_num:49/512, score: 4484155.95, best_score: 1586524.15
    iter_num:50/512, score: 4489746.81, best_score: 1586524.15
    iter_num:51/512, score: 4454786.47, best_score: 1586524.15
    iter_num:52/512, score: 4799279.22, best_score: 1586524.15
    iter_num:53/512, score: 5080819.13, best_score: 1586524.15
    iter_num:54/512, score: 5182441.8, best_score: 1586524.15
    iter_num:55/512, score: 5286231.47, best_score: 1586524.15
    iter_num:56/512, score: 5806190.28, best_score: 1586524.15
    iter_num:57/512, score: 4560688.84, best_score: 1586524.15
    iter_num:58/512, score: 4361924.59, best_score: 1586524.15
    iter_num:59/512, score: 4716026.53, best_score: 1586524.15
    iter_num:60/512, score: 5132246.5, best_score: 1586524.15
    iter_num:61/512, score: 5231600.3, best_score: 1586524.15
    iter_num:62/512, score: 5315561.87, best_score: 1586524.15
    iter_num:63/512, score: 5845952.47, best_score: 1586524.15
    iter_num:64/512, score: 6317477.6, best_score: 1586524.15
    iter_num:65/512, score: 4212308.12, best_score: 1586524.15
    iter_num:66/512, score: 3949337.4, best_score: 1586524.15
    iter_num:67/512, score: 4316709.94, best_score: 1586524.15
    iter_num:68/512, score: 4380067.93, best_score: 1586524.15
    iter_num:69/512, score: 1915089.04, best_score: 1586524.15
    iter_num:70/512, score: 1679962.49, best_score: 1586524.15
    iter_num:71/512, score: 1603879.11, best_score: 1586524.15
    iter_num:72/512, score: 1567746.14, best_score: 1567746.14
    iter_num:73/512, score: 3035725.76, best_score: 1567746.14
    iter_num:74/512, score: 3036831.84, best_score: 1567746.14
    iter_num:75/512, score: 3032548.29, best_score: 1567746.14
    iter_num:76/512, score: 3029802.63, best_score: 1567746.14
    iter_num:77/512, score: 3036503.39, best_score: 1567746.14
    iter_num:78/512, score: 3032374.92, best_score: 1567746.14
    iter_num:79/512, score: 3038384.33, best_score: 1567746.14
    iter_num:80/512, score: 3034581.13, best_score: 1567746.14
    iter_num:81/512, score: 3798687.9, best_score: 1567746.14
    iter_num:82/512, score: 3798110.53, best_score: 1567746.14
    iter_num:83/512, score: 3808543.6, best_score: 1567746.14
    iter_num:84/512, score: 3802603.55, best_score: 1567746.14
    iter_num:85/512, score: 3799012.04, best_score: 1567746.14
    iter_num:86/512, score: 3801946.96, best_score: 1567746.14
    iter_num:87/512, score: 3805475.43, best_score: 1567746.14
    iter_num:88/512, score: 3801301.24, best_score: 1567746.14
    iter_num:89/512, score: 4205749.72, best_score: 1567746.14
    iter_num:90/512, score: 4220723.3, best_score: 1567746.14
    iter_num:91/512, score: 4230704.93, best_score: 1567746.14
    iter_num:92/512, score: 4205829.69, best_score: 1567746.14
    iter_num:93/512, score: 4197967.7, best_score: 1567746.14
    iter_num:94/512, score: 4208087.51, best_score: 1567746.14
    iter_num:95/512, score: 4198941.41, best_score: 1567746.14
    iter_num:96/512, score: 4201122.08, best_score: 1567746.14
    iter_num:97/512, score: 4463255.58, best_score: 1567746.14
    iter_num:98/512, score: 4457414.92, best_score: 1567746.14
    iter_num:99/512, score: 4452271.95, best_score: 1567746.14
    iter_num:100/512, score: 4441168.45, best_score: 1567746.14
    iter_num:101/512, score: 4429324.0, best_score: 1567746.14
    iter_num:102/512, score: 4450724.42, best_score: 1567746.14
    iter_num:103/512, score: 4453621.96, best_score: 1567746.14
    iter_num:104/512, score: 4449798.75, best_score: 1567746.14
    iter_num:105/512, score: 4148851.66, best_score: 1567746.14
    iter_num:106/512, score: 4536883.6, best_score: 1567746.14
    iter_num:107/512, score: 4591890.9, best_score: 1567746.14
    iter_num:108/512, score: 4564065.73, best_score: 1567746.14
    iter_num:109/512, score: 4480690.73, best_score: 1567746.14
    iter_num:110/512, score: 4534123.32, best_score: 1567746.14
    iter_num:111/512, score: 4648284.91, best_score: 1567746.14
    iter_num:112/512, score: 4550854.25, best_score: 1567746.14
    iter_num:113/512, score: 4542620.73, best_score: 1567746.14
    iter_num:114/512, score: 4574758.23, best_score: 1567746.14
    iter_num:115/512, score: 4522224.93, best_score: 1567746.14
    iter_num:116/512, score: 4521658.2, best_score: 1567746.14
    iter_num:117/512, score: 4419799.77, best_score: 1567746.14
    iter_num:118/512, score: 4425267.43, best_score: 1567746.14
    iter_num:119/512, score: 4499187.81, best_score: 1567746.14
    iter_num:120/512, score: 4753675.81, best_score: 1567746.14
    iter_num:121/512, score: 4637801.47, best_score: 1567746.14
    iter_num:122/512, score: 4464722.82, best_score: 1567746.14
    iter_num:123/512, score: 4462123.96, best_score: 1567746.14
    iter_num:124/512, score: 4352441.68, best_score: 1567746.14
    iter_num:125/512, score: 4510396.84, best_score: 1567746.14
    iter_num:126/512, score: 4793346.73, best_score: 1567746.14
    iter_num:127/512, score: 4356965.54, best_score: 1567746.14
    iter_num:128/512, score: 4547766.97, best_score: 1567746.14
    iter_num:129/512, score: 4478139.62, best_score: 1567746.14
    iter_num:130/512, score: 4023377.83, best_score: 1567746.14
    iter_num:131/512, score: 4486005.77, best_score: 1567746.14
    iter_num:132/512, score: 4708246.93, best_score: 1567746.14
    iter_num:133/512, score: 1923108.17, best_score: 1567746.14
    iter_num:134/512, score: 1673744.27, best_score: 1567746.14
    iter_num:135/512, score: 1602822.57, best_score: 1567746.14
    iter_num:136/512, score: 1565087.39, best_score: 1565087.39
    iter_num:137/512, score: 3038932.19, best_score: 1565087.39
    iter_num:138/512, score: 3038728.59, best_score: 1565087.39
    iter_num:139/512, score: 3042925.06, best_score: 1565087.39
    iter_num:140/512, score: 3031639.11, best_score: 1565087.39
    iter_num:141/512, score: 3036883.12, best_score: 1565087.39
    iter_num:142/512, score: 3036087.92, best_score: 1565087.39
    iter_num:143/512, score: 3042116.01, best_score: 1565087.39
    iter_num:144/512, score: 3041818.92, best_score: 1565087.39
    iter_num:145/512, score: 3805368.13, best_score: 1565087.39
    iter_num:146/512, score: 3806130.67, best_score: 1565087.39
    iter_num:147/512, score: 3798792.13, best_score: 1565087.39
    iter_num:148/512, score: 3811563.58, best_score: 1565087.39
    iter_num:149/512, score: 3789594.07, best_score: 1565087.39
    iter_num:150/512, score: 3800871.98, best_score: 1565087.39
    iter_num:151/512, score: 3801332.66, best_score: 1565087.39
    iter_num:152/512, score: 3800534.98, best_score: 1565087.39
    iter_num:153/512, score: 4204882.25, best_score: 1565087.39
    iter_num:154/512, score: 4210832.98, best_score: 1565087.39
    iter_num:155/512, score: 4203778.73, best_score: 1565087.39
    iter_num:156/512, score: 4225275.31, best_score: 1565087.39
    iter_num:157/512, score: 4214966.89, best_score: 1565087.39
    iter_num:158/512, score: 4208020.67, best_score: 1565087.39
    iter_num:159/512, score: 4204852.48, best_score: 1565087.39
    iter_num:160/512, score: 4196443.3, best_score: 1565087.39
    iter_num:161/512, score: 4434643.51, best_score: 1565087.39
    iter_num:162/512, score: 4436845.85, best_score: 1565087.39
    iter_num:163/512, score: 4471715.29, best_score: 1565087.39
    iter_num:164/512, score: 4444616.19, best_score: 1565087.39
    iter_num:165/512, score: 4458419.13, best_score: 1565087.39
    iter_num:166/512, score: 4446576.3, best_score: 1565087.39
    iter_num:167/512, score: 4450430.22, best_score: 1565087.39
    iter_num:168/512, score: 4455536.73, best_score: 1565087.39
    iter_num:169/512, score: 4148851.66, best_score: 1565087.39
    iter_num:170/512, score: 4536883.6, best_score: 1565087.39
    iter_num:171/512, score: 4583912.59, best_score: 1565087.39
    iter_num:172/512, score: 4592519.33, best_score: 1565087.39
    iter_num:173/512, score: 4493830.36, best_score: 1565087.39
    iter_num:174/512, score: 4644145.29, best_score: 1565087.39
    iter_num:175/512, score: 4524274.97, best_score: 1565087.39
    iter_num:176/512, score: 4706838.62, best_score: 1565087.39
    iter_num:177/512, score: 4542620.73, best_score: 1565087.39
    iter_num:178/512, score: 4575046.11, best_score: 1565087.39
    iter_num:179/512, score: 4469754.71, best_score: 1565087.39
    iter_num:180/512, score: 4552637.39, best_score: 1565087.39
    iter_num:181/512, score: 4672787.61, best_score: 1565087.39
    iter_num:182/512, score: 4761820.44, best_score: 1565087.39
    iter_num:183/512, score: 4565669.2, best_score: 1565087.39
    iter_num:184/512, score: 4786303.79, best_score: 1565087.39
    iter_num:185/512, score: 4638441.05, best_score: 1565087.39
    iter_num:186/512, score: 4483427.45, best_score: 1565087.39
    iter_num:187/512, score: 4592732.42, best_score: 1565087.39
    iter_num:188/512, score: 4598888.01, best_score: 1565087.39
    iter_num:189/512, score: 4712519.96, best_score: 1565087.39
    iter_num:190/512, score: 4787150.74, best_score: 1565087.39
    iter_num:191/512, score: 4541976.63, best_score: 1565087.39
    iter_num:192/512, score: 4671573.46, best_score: 1565087.39
    iter_num:193/512, score: 4677464.53, best_score: 1565087.39
    iter_num:194/512, score: 4132168.76, best_score: 1565087.39
    iter_num:195/512, score: 4463153.11, best_score: 1565087.39
    iter_num:196/512, score: 4730158.45, best_score: 1565087.39
    iter_num:197/512, score: 1928644.35, best_score: 1565087.39
    iter_num:198/512, score: 1687548.39, best_score: 1565087.39
    iter_num:199/512, score: 1611930.33, best_score: 1565087.39
    iter_num:200/512, score: 1570599.71, best_score: 1565087.39
    iter_num:201/512, score: 3045130.52, best_score: 1565087.39
    iter_num:202/512, score: 3031184.96, best_score: 1565087.39
    iter_num:203/512, score: 3040076.43, best_score: 1565087.39
    iter_num:204/512, score: 3031372.61, best_score: 1565087.39
    iter_num:205/512, score: 3033881.22, best_score: 1565087.39
    iter_num:206/512, score: 3032874.88, best_score: 1565087.39
    iter_num:207/512, score: 3035963.79, best_score: 1565087.39
    iter_num:208/512, score: 3036072.69, best_score: 1565087.39
    iter_num:209/512, score: 3808307.16, best_score: 1565087.39
    iter_num:210/512, score: 3808344.61, best_score: 1565087.39
    iter_num:211/512, score: 3807952.65, best_score: 1565087.39
    iter_num:212/512, score: 3814043.83, best_score: 1565087.39
    iter_num:213/512, score: 3803263.83, best_score: 1565087.39
    iter_num:214/512, score: 3800573.0, best_score: 1565087.39
    iter_num:215/512, score: 3796552.5, best_score: 1565087.39
    iter_num:216/512, score: 3804871.72, best_score: 1565087.39
    iter_num:217/512, score: 4214957.93, best_score: 1565087.39
    iter_num:218/512, score: 4225730.39, best_score: 1565087.39
    iter_num:219/512, score: 4222547.65, best_score: 1565087.39
    iter_num:220/512, score: 4197602.05, best_score: 1565087.39
    iter_num:221/512, score: 4197529.15, best_score: 1565087.39
    iter_num:222/512, score: 4208238.06, best_score: 1565087.39
    iter_num:223/512, score: 4192736.96, best_score: 1565087.39
    iter_num:224/512, score: 4204906.29, best_score: 1565087.39
    iter_num:225/512, score: 4430605.29, best_score: 1565087.39
    iter_num:226/512, score: 4457118.29, best_score: 1565087.39
    iter_num:227/512, score: 4468634.2, best_score: 1565087.39
    iter_num:228/512, score: 4466723.1, best_score: 1565087.39
    iter_num:229/512, score: 4452214.18, best_score: 1565087.39
    iter_num:230/512, score: 4444741.03, best_score: 1565087.39
    iter_num:231/512, score: 4464621.81, best_score: 1565087.39
    iter_num:232/512, score: 4454827.51, best_score: 1565087.39
    iter_num:233/512, score: 4148851.66, best_score: 1565087.39
    iter_num:234/512, score: 4522348.29, best_score: 1565087.39
    iter_num:235/512, score: 4574466.81, best_score: 1565087.39
    iter_num:236/512, score: 4443857.42, best_score: 1565087.39
    iter_num:237/512, score: 4471509.73, best_score: 1565087.39
    iter_num:238/512, score: 4642334.68, best_score: 1565087.39
    iter_num:239/512, score: 4550943.47, best_score: 1565087.39
    iter_num:240/512, score: 4663109.58, best_score: 1565087.39
    iter_num:241/512, score: 4536104.0, best_score: 1565087.39
    iter_num:242/512, score: 4552910.61, best_score: 1565087.39
    iter_num:243/512, score: 4390265.58, best_score: 1565087.39
    iter_num:244/512, score: 4528506.7, best_score: 1565087.39
    iter_num:245/512, score: 4660052.69, best_score: 1565087.39
    iter_num:246/512, score: 4783482.47, best_score: 1565087.39
    iter_num:247/512, score: 4608524.65, best_score: 1565087.39
    iter_num:248/512, score: 4580385.95, best_score: 1565087.39
    iter_num:249/512, score: 4610629.92, best_score: 1565087.39
    iter_num:250/512, score: 4477872.8, best_score: 1565087.39
    iter_num:251/512, score: 4480173.56, best_score: 1565087.39
    iter_num:252/512, score: 4555396.32, best_score: 1565087.39
    iter_num:253/512, score: 4450830.56, best_score: 1565087.39
    iter_num:254/512, score: 4638185.49, best_score: 1565087.39
    iter_num:255/512, score: 4518723.75, best_score: 1565087.39
    iter_num:256/512, score: 4698426.56, best_score: 1565087.39
    iter_num:257/512, score: 3804480.38, best_score: 1565087.39
    iter_num:258/512, score: 4268282.69, best_score: 1565087.39
    iter_num:259/512, score: 4474767.08, best_score: 1565087.39
    iter_num:260/512, score: 4695712.65, best_score: 1565087.39
    iter_num:261/512, score: 2066568.55, best_score: 1565087.39
    iter_num:262/512, score: 1798328.51, best_score: 1565087.39
    iter_num:263/512, score: 1718113.14, best_score: 1565087.39
    iter_num:264/512, score: 1677493.29, best_score: 1565087.39
    iter_num:265/512, score: 3030723.59, best_score: 1565087.39
    iter_num:266/512, score: 3032164.28, best_score: 1565087.39
    iter_num:267/512, score: 3031574.9, best_score: 1565087.39
    iter_num:268/512, score: 3034606.78, best_score: 1565087.39
    iter_num:269/512, score: 3032137.18, best_score: 1565087.39
    iter_num:270/512, score: 3038384.93, best_score: 1565087.39
    iter_num:271/512, score: 3035216.05, best_score: 1565087.39
    iter_num:272/512, score: 3042369.71, best_score: 1565087.39
    iter_num:273/512, score: 3814587.81, best_score: 1565087.39
    iter_num:274/512, score: 3801466.23, best_score: 1565087.39
    iter_num:275/512, score: 3808185.5, best_score: 1565087.39
    iter_num:276/512, score: 3801474.44, best_score: 1565087.39
    iter_num:277/512, score: 3813608.63, best_score: 1565087.39
    iter_num:278/512, score: 3799653.96, best_score: 1565087.39
    iter_num:279/512, score: 3801994.48, best_score: 1565087.39
    iter_num:280/512, score: 3800176.38, best_score: 1565087.39
    iter_num:281/512, score: 4220169.2, best_score: 1565087.39
    iter_num:282/512, score: 4220945.98, best_score: 1565087.39
    iter_num:283/512, score: 4208607.39, best_score: 1565087.39
    iter_num:284/512, score: 4213303.79, best_score: 1565087.39
    iter_num:285/512, score: 4198132.8, best_score: 1565087.39
    iter_num:286/512, score: 4208606.84, best_score: 1565087.39
    iter_num:287/512, score: 4206922.32, best_score: 1565087.39
    iter_num:288/512, score: 4208040.59, best_score: 1565087.39
    iter_num:289/512, score: 4423106.0, best_score: 1565087.39
    iter_num:290/512, score: 4452052.23, best_score: 1565087.39
    iter_num:291/512, score: 4416665.92, best_score: 1565087.39
    iter_num:292/512, score: 4464608.74, best_score: 1565087.39
    iter_num:293/512, score: 4463795.91, best_score: 1565087.39
    iter_num:294/512, score: 4467410.76, best_score: 1565087.39
    iter_num:295/512, score: 4463802.52, best_score: 1565087.39
    iter_num:296/512, score: 4455242.36, best_score: 1565087.39
    iter_num:297/512, score: 4148851.66, best_score: 1565087.39
    iter_num:298/512, score: 4495001.39, best_score: 1565087.39
    iter_num:299/512, score: 4524188.57, best_score: 1565087.39
    iter_num:300/512, score: 4489131.49, best_score: 1565087.39
    iter_num:301/512, score: 4469130.26, best_score: 1565087.39
    iter_num:302/512, score: 4642571.24, best_score: 1565087.39
    iter_num:303/512, score: 4482965.21, best_score: 1565087.39
    iter_num:304/512, score: 4706137.88, best_score: 1565087.39
    iter_num:305/512, score: 4502898.71, best_score: 1565087.39
    iter_num:306/512, score: 4447015.82, best_score: 1565087.39
    iter_num:307/512, score: 4501679.21, best_score: 1565087.39
    iter_num:308/512, score: 4838079.19, best_score: 1565087.39
    iter_num:309/512, score: 4642370.65, best_score: 1565087.39
    iter_num:310/512, score: 4973837.35, best_score: 1565087.39
    iter_num:311/512, score: 4731744.98, best_score: 1565087.39
    iter_num:312/512, score: 5185463.94, best_score: 1565087.39
    iter_num:313/512, score: 4524278.17, best_score: 1565087.39
    iter_num:314/512, score: 4370262.52, best_score: 1565087.39
    iter_num:315/512, score: 4606693.18, best_score: 1565087.39
    iter_num:316/512, score: 4895598.39, best_score: 1565087.39
    iter_num:317/512, score: 4846425.84, best_score: 1565087.39
    iter_num:318/512, score: 5262008.81, best_score: 1565087.39
    iter_num:319/512, score: 5049088.79, best_score: 1565087.39
    iter_num:320/512, score: 5518550.95, best_score: 1565087.39
    iter_num:321/512, score: 4665353.79, best_score: 1565087.39
    iter_num:322/512, score: 4848361.53, best_score: 1565087.39
    iter_num:323/512, score: 4714119.07, best_score: 1565087.39
    iter_num:324/512, score: 4632880.19, best_score: 1565087.39
    iter_num:325/512, score: 2033177.85, best_score: 1565087.39
    iter_num:326/512, score: 1796286.84, best_score: 1565087.39
    iter_num:327/512, score: 1722224.86, best_score: 1565087.39
    iter_num:328/512, score: 1683126.49, best_score: 1565087.39
    iter_num:329/512, score: 3037396.69, best_score: 1565087.39
    iter_num:330/512, score: 3031609.65, best_score: 1565087.39
    iter_num:331/512, score: 3035182.58, best_score: 1565087.39
    iter_num:332/512, score: 3038607.94, best_score: 1565087.39
    iter_num:333/512, score: 3040550.98, best_score: 1565087.39
    iter_num:334/512, score: 3040019.2, best_score: 1565087.39
    iter_num:335/512, score: 3041352.19, best_score: 1565087.39
    iter_num:336/512, score: 3038040.8, best_score: 1565087.39
    iter_num:337/512, score: 3793032.43, best_score: 1565087.39
    iter_num:338/512, score: 3797516.55, best_score: 1565087.39
    iter_num:339/512, score: 3799218.78, best_score: 1565087.39
    iter_num:340/512, score: 3806182.42, best_score: 1565087.39
    iter_num:341/512, score: 3806864.24, best_score: 1565087.39
    iter_num:342/512, score: 3803003.26, best_score: 1565087.39
    iter_num:343/512, score: 3797582.09, best_score: 1565087.39
    iter_num:344/512, score: 3794117.84, best_score: 1565087.39
    iter_num:345/512, score: 4202587.48, best_score: 1565087.39
    iter_num:346/512, score: 4198584.29, best_score: 1565087.39
    iter_num:347/512, score: 4200236.1, best_score: 1565087.39
    iter_num:348/512, score: 4214882.01, best_score: 1565087.39
    iter_num:349/512, score: 4211679.6, best_score: 1565087.39
    iter_num:350/512, score: 4216161.37, best_score: 1565087.39
    iter_num:351/512, score: 4193295.26, best_score: 1565087.39
    iter_num:352/512, score: 4197745.96, best_score: 1565087.39
    iter_num:353/512, score: 4429340.6, best_score: 1565087.39
    iter_num:354/512, score: 4428866.2, best_score: 1565087.39
    iter_num:355/512, score: 4474490.37, best_score: 1565087.39
    iter_num:356/512, score: 4430785.17, best_score: 1565087.39
    iter_num:357/512, score: 4464234.39, best_score: 1565087.39
    iter_num:358/512, score: 4464589.26, best_score: 1565087.39
    iter_num:359/512, score: 4454773.46, best_score: 1565087.39
    iter_num:360/512, score: 4456855.67, best_score: 1565087.39
    iter_num:361/512, score: 4148851.66, best_score: 1565087.39
    iter_num:362/512, score: 4489855.79, best_score: 1565087.39
    iter_num:363/512, score: 4533358.56, best_score: 1565087.39
    iter_num:364/512, score: 4483143.31, best_score: 1565087.39
    iter_num:365/512, score: 4485751.22, best_score: 1565087.39
    iter_num:366/512, score: 4685980.92, best_score: 1565087.39
    iter_num:367/512, score: 4511327.43, best_score: 1565087.39
    iter_num:368/512, score: 4819229.17, best_score: 1565087.39
    iter_num:369/512, score: 4488024.33, best_score: 1565087.39
    iter_num:370/512, score: 4427487.9, best_score: 1565087.39
    iter_num:371/512, score: 4535305.48, best_score: 1565087.39
    iter_num:372/512, score: 4694002.4, best_score: 1565087.39
    iter_num:373/512, score: 4691091.42, best_score: 1565087.39
    iter_num:374/512, score: 4995617.02, best_score: 1565087.39
    iter_num:375/512, score: 4802503.34, best_score: 1565087.39
    iter_num:376/512, score: 4986406.74, best_score: 1565087.39
    iter_num:377/512, score: 4508013.08, best_score: 1565087.39
    iter_num:378/512, score: 4399253.34, best_score: 1565087.39
    iter_num:379/512, score: 4659092.52, best_score: 1565087.39
    iter_num:380/512, score: 4868856.88, best_score: 1565087.39
    iter_num:381/512, score: 4878009.14, best_score: 1565087.39
    iter_num:382/512, score: 5119414.85, best_score: 1565087.39
    iter_num:383/512, score: 5139305.27, best_score: 1565087.39
    iter_num:384/512, score: 5260085.79, best_score: 1565087.39
    iter_num:385/512, score: 4622589.24, best_score: 1565087.39
    iter_num:386/512, score: 4711357.34, best_score: 1565087.39
    iter_num:387/512, score: 4398379.11, best_score: 1565087.39
    iter_num:388/512, score: 4215670.4, best_score: 1565087.39
    iter_num:389/512, score: 2004648.43, best_score: 1565087.39
    iter_num:390/512, score: 1786893.93, best_score: 1565087.39
    iter_num:391/512, score: 1703209.85, best_score: 1565087.39
    iter_num:392/512, score: 1665616.73, best_score: 1565087.39
    iter_num:393/512, score: 3034151.37, best_score: 1565087.39
    iter_num:394/512, score: 3037077.08, best_score: 1565087.39
    iter_num:395/512, score: 3047002.13, best_score: 1565087.39
    iter_num:396/512, score: 3033181.47, best_score: 1565087.39
    iter_num:397/512, score: 3032148.47, best_score: 1565087.39
    iter_num:398/512, score: 3038166.1, best_score: 1565087.39
    iter_num:399/512, score: 3037791.37, best_score: 1565087.39
    iter_num:400/512, score: 3039001.32, best_score: 1565087.39
    iter_num:401/512, score: 3806683.42, best_score: 1565087.39
    iter_num:402/512, score: 3803943.24, best_score: 1565087.39
    iter_num:403/512, score: 3796974.34, best_score: 1565087.39
    iter_num:404/512, score: 3794786.69, best_score: 1565087.39
    iter_num:405/512, score: 3799680.24, best_score: 1565087.39
    iter_num:406/512, score: 3798308.55, best_score: 1565087.39
    iter_num:407/512, score: 3802623.14, best_score: 1565087.39
    iter_num:408/512, score: 3798466.74, best_score: 1565087.39
    iter_num:409/512, score: 4212900.1, best_score: 1565087.39
    iter_num:410/512, score: 4214506.15, best_score: 1565087.39
    iter_num:411/512, score: 4215902.76, best_score: 1565087.39
    iter_num:412/512, score: 4217283.86, best_score: 1565087.39
    iter_num:413/512, score: 4201061.35, best_score: 1565087.39
    iter_num:414/512, score: 4208837.09, best_score: 1565087.39
    iter_num:415/512, score: 4190049.24, best_score: 1565087.39
    iter_num:416/512, score: 4194631.63, best_score: 1565087.39
    iter_num:417/512, score: 4471441.8, best_score: 1565087.39
    iter_num:418/512, score: 4426513.93, best_score: 1565087.39
    iter_num:419/512, score: 4428442.11, best_score: 1565087.39
    iter_num:420/512, score: 4467330.95, best_score: 1565087.39
    iter_num:421/512, score: 4462887.88, best_score: 1565087.39
    iter_num:422/512, score: 4468668.07, best_score: 1565087.39
    iter_num:423/512, score: 4448358.0, best_score: 1565087.39
    iter_num:424/512, score: 4449455.7, best_score: 1565087.39
    iter_num:425/512, score: 4148851.66, best_score: 1565087.39
    iter_num:426/512, score: 4490854.4, best_score: 1565087.39
    iter_num:427/512, score: 4527277.68, best_score: 1565087.39
    iter_num:428/512, score: 4412646.73, best_score: 1565087.39
    iter_num:429/512, score: 4514294.69, best_score: 1565087.39
    iter_num:430/512, score: 4714670.15, best_score: 1565087.39
    iter_num:431/512, score: 4423756.78, best_score: 1565087.39
    iter_num:432/512, score: 4892009.97, best_score: 1565087.39
    iter_num:433/512, score: 4496588.44, best_score: 1565087.39
    iter_num:434/512, score: 4411244.74, best_score: 1565087.39
    iter_num:435/512, score: 4456503.05, best_score: 1565087.39
    iter_num:436/512, score: 4641723.72, best_score: 1565087.39
    iter_num:437/512, score: 4709398.37, best_score: 1565087.39
    iter_num:438/512, score: 4997190.79, best_score: 1565087.39
    iter_num:439/512, score: 4793312.23, best_score: 1565087.39
    iter_num:440/512, score: 5014370.33, best_score: 1565087.39
    iter_num:441/512, score: 4477560.1, best_score: 1565087.39
    iter_num:442/512, score: 4368268.12, best_score: 1565087.39
    iter_num:443/512, score: 4610906.71, best_score: 1565087.39
    iter_num:444/512, score: 4770958.11, best_score: 1565087.39
    iter_num:445/512, score: 4841021.34, best_score: 1565087.39
    iter_num:446/512, score: 5012425.9, best_score: 1565087.39
    iter_num:447/512, score: 4797770.84, best_score: 1565087.39
    iter_num:448/512, score: 4991978.94, best_score: 1565087.39
    iter_num:449/512, score: 2623970.8, best_score: 1565087.39
    iter_num:450/512, score: 2889211.36, best_score: 1565087.39
    iter_num:451/512, score: 2647781.13, best_score: 1565087.39
    iter_num:452/512, score: 2964673.87, best_score: 1565087.39
    iter_num:453/512, score: 2718053.94, best_score: 1565087.39
    iter_num:454/512, score: 2408570.55, best_score: 1565087.39
    iter_num:455/512, score: 2342640.47, best_score: 1565087.39
    iter_num:456/512, score: 2301083.56, best_score: 1565087.39
    iter_num:457/512, score: 3029507.29, best_score: 1565087.39
    iter_num:458/512, score: 3033101.96, best_score: 1565087.39
    iter_num:459/512, score: 3031292.05, best_score: 1565087.39
    iter_num:460/512, score: 3038345.79, best_score: 1565087.39
    iter_num:461/512, score: 3034039.06, best_score: 1565087.39
    iter_num:462/512, score: 3035092.93, best_score: 1565087.39
    iter_num:463/512, score: 3030329.82, best_score: 1565087.39
    iter_num:464/512, score: 3038788.2, best_score: 1565087.39
    iter_num:465/512, score: 3803686.77, best_score: 1565087.39
    iter_num:466/512, score: 3792001.12, best_score: 1565087.39
    iter_num:467/512, score: 3797949.13, best_score: 1565087.39
    iter_num:468/512, score: 3801823.42, best_score: 1565087.39
    iter_num:469/512, score: 3792197.82, best_score: 1565087.39
    iter_num:470/512, score: 3806645.72, best_score: 1565087.39
    iter_num:471/512, score: 3799818.05, best_score: 1565087.39
    iter_num:472/512, score: 3799157.95, best_score: 1565087.39
    iter_num:473/512, score: 4213611.54, best_score: 1565087.39
    iter_num:474/512, score: 4217963.82, best_score: 1565087.39
    iter_num:475/512, score: 4212801.45, best_score: 1565087.39
    iter_num:476/512, score: 4209654.43, best_score: 1565087.39
    iter_num:477/512, score: 4190253.31, best_score: 1565087.39
    iter_num:478/512, score: 4214047.79, best_score: 1565087.39
    iter_num:479/512, score: 4193140.07, best_score: 1565087.39
    iter_num:480/512, score: 4197361.91, best_score: 1565087.39
    iter_num:481/512, score: 4463216.05, best_score: 1565087.39
    iter_num:482/512, score: 4380994.14, best_score: 1565087.39
    iter_num:483/512, score: 4463041.28, best_score: 1565087.39
    iter_num:484/512, score: 4434662.09, best_score: 1565087.39
    iter_num:485/512, score: 4460535.15, best_score: 1565087.39
    iter_num:486/512, score: 4438042.53, best_score: 1565087.39
    iter_num:487/512, score: 4449181.57, best_score: 1565087.39
    iter_num:488/512, score: 4455297.57, best_score: 1565087.39
    iter_num:489/512, score: 4148851.66, best_score: 1565087.39
    iter_num:490/512, score: 4494244.99, best_score: 1565087.39
    iter_num:491/512, score: 4557846.93, best_score: 1565087.39
    iter_num:492/512, score: 4451967.93, best_score: 1565087.39
    iter_num:493/512, score: 4518934.7, best_score: 1565087.39
    iter_num:494/512, score: 4403359.24, best_score: 1565087.39
    iter_num:495/512, score: 4516622.9, best_score: 1565087.39
    iter_num:496/512, score: 4359564.14, best_score: 1565087.39
    iter_num:497/512, score: 4497912.39, best_score: 1565087.39
    iter_num:498/512, score: 4383267.57, best_score: 1565087.39
    iter_num:499/512, score: 4452265.67, best_score: 1565087.39
    iter_num:500/512, score: 4402223.39, best_score: 1565087.39
    iter_num:501/512, score: 4391312.51, best_score: 1565087.39
    iter_num:502/512, score: 4394521.96, best_score: 1565087.39
    iter_num:503/512, score: 4397881.56, best_score: 1565087.39
    iter_num:504/512, score: 4361205.99, best_score: 1565087.39
    iter_num:505/512, score: 4430393.45, best_score: 1565087.39
    iter_num:506/512, score: 4377146.26, best_score: 1565087.39
    iter_num:507/512, score: 4409049.48, best_score: 1565087.39
    iter_num:508/512, score: 4393040.73, best_score: 1565087.39
    iter_num:509/512, score: 4378160.69, best_score: 1565087.39
    iter_num:510/512, score: 4385937.05, best_score: 1565087.39
    iter_num:511/512, score: 4348946.18, best_score: 1565087.39
    iter_num:512/512, score: 4340360.13, best_score: 1565087.39
    

### 최종 모델 학습

- 파라미터 튜닝을 통해 찾은 최적의 파라미터로 전체 데이터에 대해 재학습 수행
- 이 떄, 새로 들어온 데이터에 대해서도 동일한 전처리를 하기 위해, pipeline을 함수화함


```python
def pipeline(X):
    X[biased_variables] = X[biased_variables] - X[biased_variables].min() + 1
    X[biased_variables] = np.sqrt(X[biased_variables])        
    X = pd.DataFrame(scaler.transform(X), columns = X.columns)
    X = X[best_features]
    return X
    
model = best_model(**best_paramter).fit(pipeline(X).values, Y)
```

## 적용 데이터 구성
- 새로 들어온 데이터인 submission_df 에 대해서도 모델의 입력으로 들어갈 수 있도록 전처리 수행
- 전처리된 데이터를 모델에 투입하여 출력값을 얻고, 이를 데이터프레임화하여 정리


```python
# 2019-03-01 ~ 2019-05-31
submission_df['t'] = (2019 - 2016) * 12 + 2
```


```python
# region 변수와 type_of_business 변수 부착 
submission_df['region'] = submission_df['store_id'].replace(store_to_region)
submission_df['type_of_business'] = submission_df['store_id'].replace(store_to_type_of_business)
```

#### 특징 부착


```python
submission_df['평균할부율'] = submission_df['store_id'].replace(installment_term_per_store.to_dict())
submission_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>amount</th>
      <th>t</th>
      <th>region</th>
      <th>type_of_business</th>
      <th>평균할부율</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>0.038384</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>38</td>
      <td>없음</td>
      <td>없음</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>38</td>
      <td>없음</td>
      <td>없음</td>
      <td>0.083904</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>38</td>
      <td>서울 종로구</td>
      <td>없음</td>
      <td>0.001201</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>38</td>
      <td>없음</td>
      <td>의복 액세서리 및 모조 장신구 도매업</td>
      <td>0.075077</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission_df.drop('amount', axis = 1, inplace = True)
```


```python
# t - k (k = 1, 2, 3) 시점의 부착
# submission_df의 t는 amount_sum_per_t_and_sid의 t-k과 부착되어야 하므로, amount_sum_per_t_and_sid의 t에 k를 더함

for k in range(1, 4):
    amount_sum_per_t_and_sid['t_{}'.format(k)] = amount_sum_per_t_and_sid['t'] + k    
    submission_df = pd.merge(submission_df, amount_sum_per_t_and_sid.drop('t', axis = 1), left_on = ['store_id', 't'], right_on = ['store_id', 't_{}'.format(k)])
    submission_df.rename({"amount":"{}_before_amount".format(k)}, axis = 1, inplace = True)
    submission_df.drop(['t_{}'.format(k)], axis = 1, inplace = True)
    amount_sum_per_t_and_sid.drop(['t_{}'.format(k)], axis = 1, inplace = True)
    

```


```python
# 지역 관련 변수 부착
for k in range(1, 4):
    amount_mean_per_t_and_region['t_{}'.format(k)] = amount_mean_per_t_and_region['t'] + k
    submission_df = pd.merge(submission_df, amount_mean_per_t_and_region.drop('t', axis = 1), left_on = ['region', 't'], right_on = ['region', 't_{}'.format(k)])
    submission_df.rename({"amount":"{}_before_amount_of_region".format(k)}, axis = 1, inplace = True)
    
    submission_df.drop(['t_{}'.format(k)], axis = 1, inplace = True)
    amount_mean_per_t_and_region.drop(['t_{}'.format(k)], axis = 1, inplace = True)    
```


```python
# t - k (k = 1, 2, 3) 시점의 부착
# submission_df의 t는 amount_sum_per_t_and_sid의 t-k과 부착되어야 하므로, amount_sum_per_t_and_sid의 t에 k를 더함

for k in range(1, 4):
    amount_mean_per_t_and_type_of_business['t_{}'.format(k)] = amount_mean_per_t_and_type_of_business['t'] + k
    submission_df = pd.merge(submission_df, amount_mean_per_t_and_type_of_business.drop('t', axis = 1), left_on = ['type_of_business', 't'], right_on = ['type_of_business', 't_{}'.format(k)])
    submission_df.rename({"amount":"{}_before_amount_of_type_of_business".format(k)}, axis = 1, inplace = True)
    
    submission_df.drop(['t_{}'.format(k)], axis = 1, inplace = True)
    amount_mean_per_t_and_type_of_business.drop(['t_{}'.format(k)], axis = 1, inplace = True)       
    
submission_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>t</th>
      <th>region</th>
      <th>type_of_business</th>
      <th>평균할부율</th>
      <th>1_before_amount</th>
      <th>2_before_amount</th>
      <th>3_before_amount</th>
      <th>1_before_amount_of_region</th>
      <th>2_before_amount_of_region</th>
      <th>3_before_amount_of_region</th>
      <th>1_before_amount_of_type_of_business</th>
      <th>2_before_amount_of_type_of_business</th>
      <th>3_before_amount_of_type_of_business</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>38</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>0.038384</td>
      <td>682857.142857</td>
      <td>874571.428571</td>
      <td>676000.000000</td>
      <td>9.468777e+05</td>
      <td>1.000725e+06</td>
      <td>9.886195e+05</td>
      <td>585125.0</td>
      <td>650055.952381</td>
      <td>558241.666667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>792</td>
      <td>38</td>
      <td>없음</td>
      <td>기타 미용업</td>
      <td>0.218887</td>
      <td>743214.285714</td>
      <td>871071.428571</td>
      <td>973857.142857</td>
      <td>9.468777e+05</td>
      <td>1.000725e+06</td>
      <td>9.886195e+05</td>
      <td>585125.0</td>
      <td>650055.952381</td>
      <td>558241.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1828</td>
      <td>38</td>
      <td>경기 용인시</td>
      <td>기타 미용업</td>
      <td>0.195502</td>
      <td>953000.000000</td>
      <td>816857.142857</td>
      <td>911957.142857</td>
      <td>1.801051e+06</td>
      <td>2.009936e+06</td>
      <td>1.897275e+06</td>
      <td>585125.0</td>
      <td>650055.952381</td>
      <td>558241.666667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>38</td>
      <td>경기 안양시</td>
      <td>기타 미용업</td>
      <td>0.048795</td>
      <td>660857.142857</td>
      <td>999285.714286</td>
      <td>827571.428571</td>
      <td>7.843780e+05</td>
      <td>6.421832e+05</td>
      <td>6.788446e+05</td>
      <td>585125.0</td>
      <td>650055.952381</td>
      <td>558241.666667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>192</td>
      <td>38</td>
      <td>경기 화성시</td>
      <td>기타 미용업</td>
      <td>0.100542</td>
      <td>467571.428571</td>
      <td>550571.428571</td>
      <td>399142.857143</td>
      <td>1.209348e+06</td>
      <td>1.125181e+06</td>
      <td>1.049587e+06</td>
      <td>585125.0</td>
      <td>650055.952381</td>
      <td>558241.666667</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission_X = submission_df[X.columns]
submission_X = pipeline(submission_X)

pred_Y = model.predict(submission_X)

result = pd.DataFrame({"store_id":submission_df['store_id'].values,
                      "pred_amount":pred_Y})
```

    C:\Users\choiswonspec\anaconda3\lib\site-packages\pandas\core\frame.py:3065: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self[k1] = value[k2]
    


```python
result.sort_values(by = 'store_id')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>pred_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5.168055e+06</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1.315465e+06</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>1.564663e+06</td>
    </tr>
    <tr>
      <th>612</th>
      <td>4</td>
      <td>5.263719e+06</td>
    </tr>
    <tr>
      <th>1187</th>
      <td>5</td>
      <td>4.485424e+06</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>609</th>
      <td>2132</td>
      <td>3.245156e+06</td>
    </tr>
    <tr>
      <th>610</th>
      <td>2133</td>
      <td>1.507945e+06</td>
    </tr>
    <tr>
      <th>1186</th>
      <td>2134</td>
      <td>1.180110e+06</td>
    </tr>
    <tr>
      <th>611</th>
      <td>2135</td>
      <td>2.155392e+06</td>
    </tr>
    <tr>
      <th>1463</th>
      <td>2136</td>
      <td>9.008675e+06</td>
    </tr>
  </tbody>
</table>
<p>1967 rows × 2 columns</p>
</div>




```python

```
