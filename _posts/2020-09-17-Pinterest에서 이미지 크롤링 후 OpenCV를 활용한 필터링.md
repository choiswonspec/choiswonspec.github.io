# Pinterest 이미지 크롤링 후 OpenCV 필터링


```python
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager # webdrv-manager 패키지 다운로드
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.keys import Keys
import numpy as np
import pandas as pd
import urllib.request
import requests
from tqdm.notebook import tqdm
import os
from datetime import datetime
import openpyxl

# Pinterest 사이트 접속
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get('https://www.pinterest.co.kr')
driver.implicitly_wait(1)
```


```python
# 로그인 
driver.find_element_by_xpath('//*[@id=\"__PWS_ROOT__\"]/div[1]/div/div/div/div[1]/div[1]/div[2]/div[2]/button/div').click()
driver.implicitly_wait(1)
time.sleep(1)

```


```python
my_id = 'csw31915@daum.net'

my_password = 'welcome2019!'

driver.find_element_by_xpath('//*[@id=\"email\"]').send_keys(my_id)
driver.find_element_by_xpath('//*[@id=\"password\"]').send_keys(my_password)
driver.find_element_by_xpath('//*[@id=\"__PWS_ROOT__\"]/div[1]/div/div/div/div[1]/div[2]/div[2]/div/div/div/div[1]/div/div/div/div[3]/form/div[5]/button/div').click()
time.sleep(2)
```


```python
# keyword, color_name, pk 가져오기
import pandas as pd
df = pd.read_excel("./new_keycolor_query_200907.xlsx", header=1)

```


```python
df.info()
```

* 검색어 ( search_query ) -> google_query
* kl_name (키토크) -> color_name
* 이미지 저장 path -> direct + 파일명
* 이미지 url -> 
* category_uid -> sheet 이름


```python
'''################################## 코드 끊겨서 다시 돌릴 때
df_ = df.loc[545:].reset_index(drop=True)
df_
#df_test = df_test[df_test['search_query_2'].notnull()]
#df_211_236[df_211_236['search_query_2'].isnull()]
'''
```


```python
# 경고 무시
import warnings

warnings.filterwarnings(action='ignore') 


# 파일명에 넣을 출처 이름
resource = 'pinterest'

#### 폴더명에 넣을 현재 날짜 
#date = datetime.today().strftime("%y%m%d")
date = '200904'
filename = 'use_openpyxl.xlsx'


for df in tqdm([df_]):#, df_62, df_64]):  # 각 엑셀 시트를 가져와서 반복문 
    
    #### 검색을 할 때는 언더바 _ 가 있어야 한다.
    df['search_query_2'] = df['search_query_2'].str.replace(" ", "_")
    df['kl_name'] = df['kl_name'].str.replace(" ","_")

    # 폴더 명에 넣을 sheet이름은 df['from'] 에서 가져오면 된다.
    
    
    for idx in tqdm(range(len(df))):
        # 파일명에 넣을 google_query.. 실제 검색할 내용
        word = df['search_query_2'][idx]
        
        where = df['category_uid'][idx]
        
        # 파일명에 넣을 color_name
        color = df['kl_name'][idx]
        
        # 파일명에 넣을 pk
        pk = df['kl_pk'][idx]
        
        # 사이트 접속
        driver.get('https://www.pinterest.co.kr')
        time.sleep(1.5)
        
        # 검색어에 입력후 검색
        a = driver.find_element_by_xpath('//*[@id="searchBoxContainer"]/div/div/div[2]/input')
        a.send_keys(word)
        a.send_keys(Keys.ENTER)
        time.sleep(1.5)
        
        
         # 스크롤 내리기... pinterest 사이트 특성상 스크롤을 내릴 때마다 html 수집하여 이미지 url을 수집해야한다.. 
        b = driver.find_element_by_tag_name('body')
        
        image_list = []
        escape_num = 0
        while len(image_list)<1000 and escape_num < 60:
            #for i in range(3):
            b.send_keys(Keys.END)
            time.sleep(1.5)

            # 해당 페이지 html 코드에서 이미지 url가져오기
            html = driver.page_source
            soup = BeautifulSoup(html, 'lxml')
            k = soup.find_all("img", {"class" : "hCL kVc L4E MIw"}) 
            
            time.sleep(1)

            for j in range(len(k)):
                image_list.append(k[j]['src'])
            


            image_list = list(set(image_list))
            time.sleep(0.1)
            escape_num += 1
            time.sleep(1)
            
        
        # 고화질로 얻기 위해 url 의 일부를 변경
        temp = pd.DataFrame(image_list)
        temp[0] = temp[0].str.replace("236x","564x")
        url_list = temp[0].to_list()
        
        # 폴더 만들기
        os.makedirs(f'./{date}/{resource}', exist_ok = True)
        #u = os.path.join('/Users/choiswonspec/pinterest',word)
        #os.mkdir(u)
        
        wb = openpyxl.load_workbook(filename, data_only=True)
        ws = wb.active
        count = 0
        for url in url_list:
            
            
            try:
                res = requests.get(url, verify=False, stream=True)
                rawdata = res.raw.read()
                direct = f'./{date}/{resource}'
                with open(os.path.join(direct, str(pk) +"_"+ color +"_"+ str(where) +"_"+ str(count) + '.jpg'), 'wb') as f:
                    f.write(rawdata)
                    
                    time.sleep(0.1)
                ###### db data 저장
                path_full = f'{date}/{resource}/' + str(pk) +"_"+ color +"_"+ str(where) +"_"+ str(count) + '.jpg'
                count += 1
                
                ws.append([word, color, path_full, url, where, pk])

               
               
 
                time.sleep(0.1)
                
     
            except Exception as e:
                print('Failed to write rawdata.')

            if count == 1000:
                break
        wb.save(filename)
        wb.close()
        time.sleep(1)

    

```


```python
import pandas as pd

cv = pd.read_excel('./use_openpyxl.xlsx')
cv['is_use'] = 1

import cv2
from tqdm import tqdm
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
for i in tqdm(range(len(cv))):
    try:
        image = cv2.imread(cv['path'][i])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 2) 
        # 1 또는 -1 로 분류하기
        if len(faces) >= 1:
            cv['is_use'][i] = 1
        else:
            cv['is_use'][i] = -1
    except:
        cv['is_use'][i] = 0

cv
```

      0%|          | 0/39400 [00:00<?, ?it/s]<ipython-input-1-81ecec3117d3>:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      cv['is_use'][i] = -1
      0%|          | 2/39400 [00:00<55:31, 11.83it/s]<ipython-input-1-81ecec3117d3>:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      cv['is_use'][i] = 1
    100%|██████████| 39400/39400 [1:38:58<00:00,  6.63it/s]  
    




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
      <th>search_query</th>
      <th>kl_name</th>
      <th>path</th>
      <th>url_column</th>
      <th>category_uid</th>
      <th>kl_pk</th>
      <th>is_use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wide_awake_skin_makeup_look</td>
      <td>wide_awake_look</td>
      <td>200904/pinterest/2838_wide_awake_look_62_0.jpg</td>
      <td>https://i.pinimg.com/564x/d8/32/bf/d832bfab317...</td>
      <td>62</td>
      <td>2838</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wide_awake_skin_makeup_look</td>
      <td>wide_awake_look</td>
      <td>200904/pinterest/2838_wide_awake_look_62_1.jpg</td>
      <td>https://i.pinimg.com/564x/af/5e/a5/af5ea59713e...</td>
      <td>62</td>
      <td>2838</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>wide_awake_skin_makeup_look</td>
      <td>wide_awake_look</td>
      <td>200904/pinterest/2838_wide_awake_look_62_2.jpg</td>
      <td>https://i.pinimg.com/564x/30/ff/d5/30ffd5846c1...</td>
      <td>62</td>
      <td>2838</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wide_awake_skin_makeup_look</td>
      <td>wide_awake_look</td>
      <td>200904/pinterest/2838_wide_awake_look_62_3.jpg</td>
      <td>https://i.pinimg.com/564x/6c/a2/4c/6ca24cee2ad...</td>
      <td>62</td>
      <td>2838</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>wide_awake_skin_makeup_look</td>
      <td>wide_awake_look</td>
      <td>200904/pinterest/2838_wide_awake_look_62_4.jpg</td>
      <td>https://i.pinimg.com/564x/40/d8/82/40d88237990...</td>
      <td>62</td>
      <td>2838</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39395</th>
      <td>seamless_lip_makeup_look_inspo</td>
      <td>seamless</td>
      <td>200904/pinterest/24342_seamless_64_78.jpg</td>
      <td>https://i.pinimg.com/564x/d5/3b/75/d53b75d4821...</td>
      <td>64</td>
      <td>24342</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>39396</th>
      <td>seamless_lip_makeup_look_inspo</td>
      <td>seamless</td>
      <td>200904/pinterest/24342_seamless_64_79.jpg</td>
      <td>https://i.pinimg.com/564x/30/75/0b/30750b63264...</td>
      <td>64</td>
      <td>24342</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>39397</th>
      <td>seamless_lip_makeup_look_inspo</td>
      <td>seamless</td>
      <td>200904/pinterest/24342_seamless_64_80.jpg</td>
      <td>https://i.pinimg.com/564x/48/ac/58/48ac585d13f...</td>
      <td>64</td>
      <td>24342</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>39398</th>
      <td>seamless_lip_makeup_look_inspo</td>
      <td>seamless</td>
      <td>200904/pinterest/24342_seamless_64_81.jpg</td>
      <td>https://i.pinimg.com/564x/cb/3a/db/cb3adb81f26...</td>
      <td>64</td>
      <td>24342</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>39399</th>
      <td>seamless_lip_makeup_look_inspo</td>
      <td>seamless</td>
      <td>200904/pinterest/24342_seamless_64_82.jpg</td>
      <td>https://i.pinimg.com/564x/98/f6/fa/98f6fa22b99...</td>
      <td>64</td>
      <td>24342</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>39400 rows × 7 columns</p>
</div>




```python
cv.to_csv("./hhhh.csv", index=False, encoding='utf-8')
```
