import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# 이 코드는 멀티프로세싱(GPU 학습) 시 필수입니다.
if __name__ == "__main__":
    # 1. 학습된 'best.pt' 모델 로드
    #    (이전 학습이 train6 폴더에 저장되었다고 가정)
    model_path = r'runs\detect\train6\weights\best.pt'
    model = YOLO(model_path)

    # 2. 추적할 원본 영상 파일 경로 (수정된 부분)
    #    '56.mp4'가 아니면, '56.mkv' 등으로 꼭 수정해 주세요!
    source_video = r'C:\Users\AISW-509-182\Desktop\MLB_Project\60.mp4'

    # 3. 추적(Track) 실행
    results = model.track(
        source=source_video,
        save=True,  # 궤적이 그려진 결과 영상을 파일로 저장
        workers=0,
        batch=8
    )

    print("추적이 완료되었습니다!")
    print(f"결과 영상은 runs\\detect\\track 폴더에 저장되었습니다.")