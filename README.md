# ⚾ YOLOv8 Baseball Pitch Tracking & ABS System
> **Computer Vision Project (2025.10 ~ 2025.12)**
> **Topic:** YOLOv8 기반 야구 투구 추적 및 자동 투구 판정 시스템(ABS) 구현

## 📖 Introduction (서비스 소개)
**"0.01초의 승부, AI 심판으로 구현하다"**

본 프로젝트는 시속 140km가 넘는 빠른 야구공을 일반 카메라 환경에서도 정밀하게 추적하고, **스트라이크(STRIKE)**와 **볼(BALL)** 여부를 자동으로 판정하는 컴퓨터 비전 시스템입니다.  
단순히 AI 모델에만 의존하는 것이 아니라, **선형 보간법(Linear Interpolation)** 알고리즘을 직접 구현하여 고속 물체 추적 시 발생하는 **'터널링 현상(Tunneling Effect)'**을 극복했습니다.

## 🛠 Tech Stack (기술 스택)
- **Language:** Python 3.x
- **AI / Model:** YOLOv8s (Custom Training), Ultralytics
- **Computer Vision:** OpenCV (Visual Trail Visualization)
- **Algorithm:** NumPy (Linear Interpolation, Vector Calculation)
- **Environment:** Anaconda, PyCharm

## 📷 Key Features & Results (주요 기능 및 결과)

### 1. 스트라이크(STRIKE) 판정
> 공이 스트라이크 존을 통과하면 **파란색 혜성 꼬리**와 함께 STRIKE 판정
<img width="377" height="264" alt="image" src="https://github.com/user-attachments/assets/7d843b24-1541-4b6a-894c-4edecc31464a" />


### 2. 볼(BALL) / 폭투 판정
> 공이 존을 벗어나거나 바닥으로 떨어지면 **초록색 꼬리**와 함께 BALL 판정
<img width="377" height="346" alt="image" src="https://github.com/user-attachments/assets/4460db47-85f2-4bca-af6c-9aa9db2e4603" />


### 3. 핵심 알고리즘 : 선형 보간법 (Anti-Tunneling)
> 빠른 공이 프레임 사이에서 사라지는 현상을 막기 위해, 이전/현재 좌표 사이를 **20등분 하여 가상의 좌표를 생성**하는 알고리즘 적용
<img width="360" height="317" alt="image" src="https://github.com/user-attachments/assets/6cf259cd-e943-4aed-a904-e827df6c2c85" />


## 🏗 Core Logic (핵심 로직)
1. **Input:** 야구 경기 영상 (MP4) + 사용자 ROI 설정
2. **AI Layer:** YOLOv8s 객체 탐지 & 좌표 추출
3. **Logic Layer:**
    - **노이즈 필터링:** 관중석이나 포수 미트 등 오인식 객체 제거 (좌표 기반)
    - **선형 보간 (Linear Interpolation):** 프레임 간 터널링 현상 방지
    - **계층적 판정:** Box Hit(스트라이크) → Floor Check(볼) → Missing Check(볼)
4. **Output:** OpenCV 시각화 (Visual Trail) & 판정 텍스트 오버레이

## 💡 Troubleshooting (문제 해결)
**Q. 공이 너무 빨라서 프레임 사이에서 사라진다면?** 👉 **선형 보간법**을 도입하여 프레임 간 20개의 가상 좌표를 수학적으로 생성, 존을 스치듯 지나가는 공까지 100% 감지 성공.

**Q. 관중이나 포수의 미트를 공으로 착각한다면?** 👉 투구 시작점(`start_y`)을 기준으로 터무니없는 위치의 객체는 제외하고, 30px 이상 이동 시에만 추적하는 **좌표 필터링** 로직 적용.

## 🚀 How to Run (실행 방법)
```bash
# 1. Clone this repository
git clone [https://github.com/](https://github.com/)[본인깃허브아이디]/[저장소이름].git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the tracking system
python project_abs.py
