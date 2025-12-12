import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- (원래 코드) ---
from ultralytics import YOLO
import torch

# 이 코드는 멀티프로세싱(GPU 학습) 시 필수입니다.
if __name__ == "__main__":
    # 1. 기본 모델 로드
    model = YOLO('yolov8s.pt')

    # 2. data.yaml 파일의 '절대 경로'를 변수로 정의
    data_path = r'C:\Users\AISW-509-182\Desktop\ball.v4i.yolov8\data.yaml'

    # 3. 학습(Train) 실행
    results = model.train(
        data=data_path,
        model='yolov8n.pt',
        epochs=200,
        patience=50,
        imgsz=1024,
        workers=0,
        batch=4
    )

    # 4. (선택적) 학습 결과 확인
    print("학습이 완료되었습니다!")
    print(f"최종 모델은 {results.save_dir}에 저장되었습니다.")