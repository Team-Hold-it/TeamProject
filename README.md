# 🎉✨ Welcome `멈춰!(Hold it!)` Project 😀


<div style="text-align: center;">
<img style="max-width: 70%;"
    src="https://leeyeonjun85.github.io/assets/images/etc/project-2822430_1920.jpg"
    alt="image">
</div>


## 목차
- [🎉✨ Welcome `멈춰!(Hold it!)` Project 😀](#-welcome-Hold it!-project-)
  - [목차](#목차)
  - [📜 스토리](#-스토리)
  - [프로젝트 목표](#프로젝트-목표)
    - [목표1](#목표1)
    - [목표2](#목표2)
    - [목표3](#목표3)
  - [AI 모델 1 : 전복 연령 예측](#ai-모델-1--전복-연령-예측)
  - [AI 모델 2 : 펄사 별(Pulsar Star) 관측](#ai-모델-2--펄사-별pulsar-star-관측)
  - [AI 모델 3 : 철판 불량 검출](#ai-모델-3--철판-불량-검출)
  - [Baseline Code](#baseline-code)
  - [멤버 소개](#멤버-소개)
  - [git 관련 bash 명령어](#git-관련-bash-명령어)
  - [문제성, 김동현, 권순범, 이연준 팀](#문제성-김동현-권순범-이연준-팀)
  - [링크](#링크)


<br><br><hr>


## 📜 스토리
<div>
    <p>우리 회사 <code>AI Code</code>는 다양한 인공지능 모델을 개발하여 필요한 회사에 제공합니다.</p>
    <p>최근 AI Code는 3개 회사에서 <b>인공지능 모델 업그레이드</b> 업무를 수주하였습니다.</p>
    <p>모델업그레이드 뿐만 아니라 사용자들이 업그레이드 된 모델을 직접 경험할 수 있도록 <b>웹서비스를 제공</b>해야 합니다.</p>
</div>


<br><br><hr>

## 프로젝트 목표
### 목표1
- 제공받은 **세 가지 데이터**에 대하여 **탐색적 데이터 분석을 수행**합니다. 
- 분석 이후, **각 데이터에 대한 AI Model 성능 향상을 목적으로 ‘인사이트 도출’, ‘지금 바로 적용 가능한 제언’ 과 ‘추후 적용 가능한 제언’ 을 제공**합니다.

### 목표2
- 제공받은 세 가지 데이터에 대하여 사전에 구축된 AI Model의 성능 고도화 실험 및 적용 작업을 진행합니다.( ※ Baseline AI Model 코드는 제공됩니다. )
- 사전에 구축된 AI Model의 성능에 대한 자세한 내용은 아래 [AI 모델 1 : 전복 연령 예측](#ai-모델-1--전복-연령-예측), [AI 모델 2 : 펄사 별(Pulsar Star) 관측](#ai-모델-2--펄사-별pulsar-star-관측), [AI 모델 3 : 철판 불량 검출](#ai-모델-3--철판-불량-검출) 참조

### 목표3
- AI Model 은 프로그램화 하여 유저 친화적으로 사용될 수 있도록 로직을 수정합니다.
- 프로그램은 ipynb, py 타입의 파일 형식으로 구축하거나 혹은 Python Web 라이브러리 Flask, Django를 활용하여 구축합니다.
- 사전 구축된 프로그램의 데이터 전달 방식
```python
def load_dataset():
    with open('dataset.csv')as csvfile:
        csvreader= csv.reader(csvfile)
        next(csvreader,None)
        rows= []
        for rowin csvreader:
            rows.append(row)
    ...
```
- 사전 구축된 프로그램 실행 방식
  - [회귀]
    ```python
    # 회귀문제 단층 퍼셉트론 ANN 모델 불러오기 
    %run /content/AnnModel1.ipynb
    # 다층 퍼셉트론 기능 추가 
    %run /content/mlp.ipynb

    # 두개의 은닉계층에 각 5개, 3개의 노드를 갖는 모델 구현
    set_hidden(info=[5,3])
    # 메서드 동작
    main(epoch_count=10, mb_size=10, report=1, train_ratio=0.6)
    ```
    ```python
    [안내] 은닉 계층 2개를 갖는 다층 퍼셉트론이 적용됩니다.
    [Epoch 1] Train - Loss = 9.880, Accuracy = 0.775 / Test - Accuracy = 0.827
    [Epoch 2] Train - Loss = 7.074, Accuracy = 0.805 / Test - Accuracy = 0.845
    [Epoch 3] Train - Loss = 6.356, Accuracy = 0.814 / Test - Accuracy = 0.824
    [Epoch 4] Train - Loss = 6.075, Accuracy = 0.816 / Test - Accuracy = 0.854
    [Epoch 5] Train - Loss = 6.223, Accuracy = 0.816 / Test - Accuracy = 0.849
    [Epoch 6] Train - Loss = 5.910, Accuracy = 0.819 / Test - Accuracy = 0.854
    [Epoch 7] Train - Loss = 5.671, Accuracy = 0.824 / Test - Accuracy = 0.854
    [Epoch 8] Train - Loss = 5.549, Accuracy = 0.826 / Test - Accuracy = 0.761
    [Epoch 9] Train - Loss = 5.663, Accuracy = 0.825 / Test - Accuracy = 0.852
    [Epoch 10] Train - Loss = 5.459, Accuracy = 0.830 / Test - Accuracy = 0.832

    ========================================  FINAL TEST REPORT ======================================== 

    ► Final Accuracy = 0.832
    ```

    - [이진 판단]
    ```python
    # 이진 판단 문제해결 모델 불러오기 및 다층 퍼셉트론 모델 지원
    %run /content/AnnModel2.ipynb
    %run /content/mlp.ipynb

    # 2개의 은닉계층을 갖는 경우 
    set_hidden(info=[5,3])
    binary_main(epoch_count=10, mb_size=10, report=1, train_ratio=0.6)
    ```
    ```python
    [안내] 은닉 계층 2개를 갖는 다층 퍼셉트론이 적용됩니다.
    [Epoch 1] Train Loss = 0.707 | Test Acc = 0.856, Precision = 0.856, Recall = 0.853, F1 = 0.855
    [Epoch 2] Train Loss = 0.444 | Test Acc = 0.854, Precision = 0.820, Recall = 0.907, F1 = 0.861
    [Epoch 3] Train Loss = 0.382 | Test Acc = 0.877, Precision = 0.855, Recall = 0.906, F1 = 0.880
    [Epoch 4] Train Loss = 0.360 | Test Acc = 0.895, Precision = 0.900, Recall = 0.887, F1 = 0.894
    [Epoch 5] Train Loss = 0.368 | Test Acc = 0.894, Precision = 0.885, Recall = 0.906, F1 = 0.895
    [Epoch 6] Train Loss = 0.358 | Test Acc = 0.869, Precision = 0.925, Recall = 0.802, F1 = 0.859
    [Epoch 7] Train Loss = 0.345 | Test Acc = 0.894, Precision = 0.905, Recall = 0.879, F1 = 0.892
    [Epoch 8] Train Loss = 0.339 | Test Acc = 0.894, Precision = 0.891, Recall = 0.897, F1 = 0.894
    [Epoch 9] Train Loss = 0.334 | Test Acc = 0.883, Precision = 0.929, Recall = 0.827, F1 = 0.875
    [Epoch 10] Train Loss = 0.334 | Test Acc = 0.891, Precision = 0.920, Recall = 0.854, F1 = 0.886

    ========================================  FINAL TEST REPORT ======================================== 

    ► Final Acc = 0.891 Precision = 0.920, Recall = 0.854, F1 = 0.886
    ```

    - [다중 분류]
    ```python
    # 다중 분류 문제해결 모델 불러오기 및 다층 퍼셉트론 모델 지원 
    %run /content/AnnModel3.ipynb
    %run /content/mlp.ipynb

    set_hidden(info=[10,9,8])
    multiple_main(epoch_count=10, mb_size=10, report=1, train_ratio=0.6)
    ```
    ```python
    [안내] 은닉 계층 3개를 갖는 다층 퍼셉트론이 적용됩니다.
    [Epoch 1] Train - Loss = 2.578, Accuracy = 0.326 / Test - Accuracy = 0.366
    [Epoch 2] Train - Loss = 1.851, Accuracy = 0.334 / Test - Accuracy = 0.366
    [Epoch 3] Train - Loss = 1.810, Accuracy = 0.334 / Test - Accuracy = 0.366
    [Epoch 4] Train - Loss = 1.781, Accuracy = 0.334 / Test - Accuracy = 0.366
    [Epoch 5] Train - Loss = 1.761, Accuracy = 0.334 / Test - Accuracy = 0.366
    [Epoch 6] Train - Loss = 1.747, Accuracy = 0.334 / Test - Accuracy = 0.366
    [Epoch 7] Train - Loss = 1.737, Accuracy = 0.334 / Test - Accuracy = 0.366
    [Epoch 8] Train - Loss = 1.730, Accuracy = 0.334 / Test - Accuracy = 0.366
    [Epoch 9] Train - Loss = 1.724, Accuracy = 0.334 / Test - Accuracy = 0.366
    [Epoch 10] Train - Loss = 1.720, Accuracy = 0.334 / Test - Accuracy = 0.366

    ========================================  FINAL TEST REPORT ======================================== 

    ► Final Accuracy = 0.366
    ```


<br><br><hr>


## AI 모델 1 : 전복 연령 예측
<div>
<h5>1. 데이터</h5>
<p>• 독립변수 : 성별, 길이, 둘레 등 8개 특성</p>
<p>• 종속변수 : 전복 연령</p>
<br>
<h5>2. 기본 모델 성능 : 회귀</h5>
<p>• Epoch : 10 , lr : 0.001</p>
<p>• Train Loss / Accuracy : 5.863 / 0.824</p>
<p>• Test Accuracy : 0.827</p>

- 성능 평가 수식
> $$Loss : \frac{\Sigma^M_{i=1}\Sigma^N_{j=1}Square_{ij}}{MN}$$  
>    - $M = MiniBatch\; Size$
>    - $N = Output\;Count$
>    - $Square_{ij} = (y^{(i)}-f(x^{(i)}))^2$    
>$$Acc : 1- |\frac{\hat{y}-y}{y}|$$  

<h5>3. 자세한 탐색적 데이터분석 바로가기</h5>
<div class="text-center indent_0"><a class="btn btn-outline-primary"
    href="http://leeyj85.shop/GPTeachersDay/eda_abalone/">
    EDA 전복 : 바로가기
</a></div>
</div>


<br><br><hr>


## AI 모델 2 : 펄사 별(Pulsar Star) 관측
<div>
<h5>1. 데이터</h5>
<p>• 독립변수 : 통합프로파일(Integrated Profile) 평균, 표준편차 등 8개 특성</p>
<p>• 종속변수 : 펄사 별(Pulsar Star) 여부</p>
<br>
<h5>2. 기본 모델 성능 : 이진 판단</h5>
<p>• Epoch : 10 , lr : 0.001</p>
<p>• Train Loss : 1.014</p>
<p>• Test<br>
    &nbsp;&nbsp;Acc = 0.976, Precision = 0.926, Recall = 0.789, F1 = 0.852</p>

- $Loss(Sigmoid\;Cross\;Entropy)$

![bi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/bi_loss.png?raw=true)

>- $Precision = \frac{True \ Positive}{True \ Positive + False \ Positive}$
>- $Recall = \frac{True \ Positive}{True \ Positive + False \ Negative}$
>- $Accuracy = \frac{True \ Positive + True \ Negative}{True \ Positive + True \ Negative + False \ Positive + False \ Negative}$

<h5>3. 자세한 탐색적 데이터분석 바로가기</h5>
<div class="text-center indent_0"><a class="btn btn-outline-danger"
    href="http://leeyj85.shop/GPTeachersDay/eda_star/">
    EDA 별 : 바로가기
</a></div>
</div>


<br><br><hr>


## AI 모델 3 : 철판 불량 검출
<div>
<h5>1. 데이터</h5>
<p>• 독립변수 : 불량 이미지 픽셀넓이, 휘도, 철판 두께 등 27개 특성</p>
<p>• 종속변수 : Pastry, Z_Scratch, 오염 등 7가지 불량</p>
<br>
<h5>2. 기본 모델 성능 : 다중 분류</h5>
<p>• Epoch : 10 , lr : 0.001</p>
<p>• Train Loss : 16.039</p>
<p>• Train Accuracy : 0.303</p>
<p>• Test Accuracy : 0.412</p>

- $Loss(Softmax\;Cross\;Entropy)$
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/multi_loss.png?raw=true)

- $Acc$ : Softmax 활성화 함수 기반의 일치여부를 비교합니다.
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/multi_acc.png?raw=true)


<h5>3. 자세한 탐색적 데이터분석 바로가기</h5>
<div class="text-center indent_0"><a class="btn btn-outline-success"
    href="http://leeyj85.shop/GPTeachersDay/eda_steel/">
    EDA 철판 : 바로가기
</a></div>
</div>


<br><br><hr>


## Baseline Code
- C사의 AI R&D 그룹에서 개발된 Numpy 기반의 레거시 코드 입니다.
- 프로그램 동작 방식
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/AI_progress.png?raw=true)

- AI Model 구현을 위한 레거시 코드 / [회귀] 
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/legacy_1_1.png?raw=true)
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/legacy_1_2.png?raw=true)

- AI Model 구현을 위한 레거시 코드 / [이진 판단_V1] 
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/legacy_2_1.png?raw=true)
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/legacy_2_2.png?raw=true)

- AI Model 구현을 위한 레거시 코드 / [이진 판단_V2]
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/legacy_2_3.png?raw=true)
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/legacy_2_4.png?raw=true)

- AI Model 구현을 위한 레거시 코드 / [다중 분류]
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/legacy_3_1.png?raw=true)
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/legacy_3_2.png?raw=true)

- AI Model 구현을 위한 레거시 코드 / [MLP]
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/legacy_mlp_1.png?raw=true)
![multi_loss](https://github.com/GPTeachersDay/TeamProject1/blob/main/colabo/img/legacy_mlp_2.png?raw=true)





<br><br><hr>


## 멤버 소개

<table style="width : 80%; margin : auto;">
    <tbody style="width : 100%; display : table;">
        <tr style="border-bottom : 3px solid gray; background-color : #88bb88;">
            <th style="width : 30%; text-align : center;">Name</th>
            <th style="width : 40%; text-align : center;">Role</th>
            <th style="width : 30%; text-align : center;">Link</th>
        </tr>
        <tr style="border-bottom : 1px solid gray;">
            <td style="width : 30%; text-align : center;">문제성</td>
            <td style="width : 40%; text-align : start;">
                • Header <br> • EDA : Pulsar Star <br> • Modeling : Abalone, Pulsar Star(SOTA) <br> • Video : Intro & Outro
            </td>
            <td style="width : 30%; text-align : center;"><a class="nav-link" href="https://github.com/JeseongMoon" target='_blank'>GitHub<i class="bi bi-github"></i></a></td>
        </tr>
        <tr style="border-bottom : 1px solid gray;">
            <td style="width : 30%; text-align : center;">김동현</td>
            <td style="width : 40%; text-align : start;">
                • Modeling : Faulty Steel(SOTA) <br> • Custom Accuracy <br> • Video : Faulty Steel <br> • Modeling : Pulsar Star
            </td>
            <td style="width : 30%; text-align : center;"><a class="nav-link" href="https://github.com/Peregrine9363" target='_blank'>GitHub<i class="bi bi-github"></i></a></td>
        </tr>
        <tr style="border-bottom : 1px solid gray;">
            <td style="width : 30%; text-align : center;">권순범</td>
            <td style="width : 40%; text-align : start;">
                • PPT : Initiative Draft <br> • Video : Pulsar Star <br> • Modeling : Abalone <br> • Optimizer 
            </td>
            <td style="width : 30%; text-align : center;"><a class="nav-link" href="https://github.com/kwonsoonbeom" target='_blank'>GitHub<i class="bi bi-github"></i></a></td>
        </tr>
        <tr style="border-bottom : 1px solid gray;">
            <td style="width : 30%; text-align : center;">이연준</td>
            <td style="width : 40%; text-align : start;">
                • EDA <br> • Engineering <br> • Modeling : Abalone(SOTA) <br> • Video : Abalone
            </td>
            <td style="width : 30%; text-align : center;"><a class="nav-link" href="https://github.com/leeyeonjun85" target='_blank'>GitHub<i class="bi bi-github"></i></a></td>
        </tr>
    </tbody>
</table>

<br><br><hr>

## git 관련 bash 명령어
```bash
# 브랜치 상태 확인
git status
# 브랜치 만들기
git branch yeonjun3
# 브랜치 삭제
git branch -d yeonjun3
# 브랜치 이동
git checkout yeonjun3
```


<br><br><hr>


## 문제성, 김동현, 권순범, 이연준 팀

```python
GPTeachers_Day = {  
    'Team_Name' : 'GPTeachers_Day',
    'Project_Name' : 'AI 소프트웨어 업그레이드 프로젝트',
    'Team_Goal' : 'GPTeacher’s Day팀의 목표는 깃헙을 이용해 **협업** 프로젝트 적극 활용하기 입니다.',
    'Members' : {  
        '문제성' : {  
            'gitid' : 'JeseongMoon',  
            'MBTI' : 'ENFJ',  
            'Role' : [Header, EDA, Modeling],  
            },  
        '김동현' : {  
            'gitid' : 'Peregrine9363',  
            'MBTI' : 'INTP',  
            'Role' : [Modeling, Custom Accuracy],  
            },  
        '권순범' : {  
            'gitid' : 'kwonsoonbeom',  
            'MBTI' : 'ISTJ'  
            'Role' : [Modeling, Optimizer],  
            },  
        '이연준' : {  
            'gitid' : 'leeyeonjun85',  
            'MBTI' : 'ISTP'  
            'Role' : [EDA, Engineering],  
            },  
        },  
    }  
```


<br><br><hr>


## 링크
- <a href="http://leeyj85.shop/GPTeachersDay" target="_blank">WEB 프로젝트 페이지</a>
- <a href="http://leeyj85.shop/GPTeachersDay/model/use1/" target="_blank">업그레이드모델 : 전복</a>
- <a href="http://leeyj85.shop/GPTeachersDay/model/use2/" target="_blank">업그레이드모델 : 맥동성</a>
- <a href="http://leeyj85.shop/GPTeachersDay/model/use3/" target="_blank">업그레이드모델 : 강철판</a>
- <a href="https://codestates.notion.site/AIB-17-Team-Project-1-2023-05-15-2023-05-25-9454e090dcdf4cf891c71c0b4bd2ba5e" target="_blank">프로젝트 노션</a>
- <a href="https://www.notion.so/9891e517ff9a473491a1d4d2f3a87221?v=d776e70e97454284b0cc4c6988a77a51" target="_blank">팀 노션페이지</a>
- <a href="https://www.notion.so/1-1-23de33f86c034ca4836fb0d45bbad632" target="_blank">1일차 노션</a>
- <a href="https://www.notion.so/1-2-20fbb27c574f409a838f22aeeab6636d" target="_blank">2일차 노션</a>
- <a href="https://www.notion.so/1-3-8df24c40ff3146aaa7f1adf8fc1a1f3a" target="_blank">3일차 노션</a>
- <a href="https://www.notion.so/1-4-f0011339e35143f7a98daff17746856e" target="_blank">4일차 노션</a>
- <a href="https://www.notion.so/1-5-74bbb5f192324074ab4042312ba97c5c" target="_blank">5일차 노션</a>
- <a href="https://www.notion.so/1-6-d71da6cae65446a8805f29ce147c5c37" target="_blank">6일차 노션</a>
- <a href="https://www.notion.so/1-7-70feca0a849544c09cf007c061084982" target="_blank">7일차 노션</a>
- <a href="https://www.notion.so/1-8-c0f4d48ebc0245c3b767df3d687acd08" target="_blank">8일차 노션</a>
- <a href="https://www.notion.so/1-9-09ea6055070d4ea59b0fd6369f5bae7b" target="_blank">9일차 노션</a>
- <a href="https://www.notion.so/1-10-1272b6eaf94d4bdf8eabf293bb1901ce" target="_blank">10일차 노션</a>
