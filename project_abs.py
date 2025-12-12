import os
import cv2
import numpy as np
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -------------------------------------------------------------
# 1. 박스 진입 확인 (스트라이크 판정용)
# -------------------------------------------------------------
def check_box_hit(p1, p2, zone):
    zx1, zy1, zx2, zy2 = zone
    padding = 15  # 스침 허용
    px1, py1 = zx1 - padding, zy1 - padding
    px2, py2 = zx2 + padding, zy2 + padding

    steps = 20
    for i in range(steps + 1):
        t = i / steps
        bx = int(p1[0] + (p2[0] - p1[0]) * t)
        by = int(p1[1] + (p2[1] - p1[1]) * t)
        if px1 <= bx <= px2 and py1 <= by <= py2:
            return True
    return False


if __name__ == "__main__":

    # ===================== [설정] =====================
    model_path = r'runs\detect\train7\weights\best.pt'
    input_folder = r"C:\Users\AISW-509-182\Desktop\MLB_Project"
    output_folder = r"runs\detect\final_results"
    # =================================================

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.MOV'))])
    print(f"총 {len(video_files)}개의 영상을 찾았습니다.")

    for i, file_name in enumerate(video_files):
        model = YOLO(model_path)
        source_path = os.path.join(input_folder, file_name)
        save_name = f"result_{file_name}".replace('.avi', '.mp4')
        output_path = os.path.join(output_folder, save_name)

        print(f"\n[{i + 1}/{len(video_files)}] '{file_name}' 처리 시작...")

        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened(): continue

        success, first_frame = cap.read()
        if not success: continue

        print("👉 마우스로 ABS 존을 그리고 [SPACE]를 누르세요!")
        r = cv2.selectROI("DRAW ABS ZONE", first_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("DRAW ABS ZONE")

        if r[2] == 0 or r[3] == 0:
            ABS_ZONE = (605, 290, 660, 370)
        else:
            ABS_ZONE = (int(r[0]), int(r[1]), int(r[0] + r[2]), int(r[1] + r[3]))

        zone_x1, zone_y1, zone_x2, zone_y2 = ABS_ZONE

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), (w, h))

        ball_trail = []
        final_decision = ""
        missing_frames = 0
        start_y = -1

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            results = model.track(frame, persist=True, verbose=False, conf=0.2)
            annotated_frame = frame.copy()

            # 존 그리기 (흰색 박스)
            cv2.rectangle(annotated_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 255, 255), 2)

            best_ball = None
            max_conf = -1

            if results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    bw, bh = x2 - x1, y2 - y1
                    conf = float(box.conf)
                    if bw < 3 or bh < 3: continue
                    if bw > 200 or bh > 200: continue
                    if conf > max_conf:
                        max_conf = conf
                        best_ball = box

            if best_ball:
                missing_frames = 0
                x1, y1, x2, y2 = best_ball.xyxy[0]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if start_y == -1: start_y = cy

                is_valid = False
                # 관중석/바닥 오인식 제거
                if start_y != -1 and (cy < start_y - 150 or cy > h - 50):
                    is_valid = False
                elif len(ball_trail) == 0:
                    is_valid = True
                else:
                    last_x, last_y = ball_trail[-1]
                    dx = abs(cx - last_x)
                    if dx < 300: is_valid = True

                if is_valid:
                    ball_trail.append((cx, cy))
                    # 꼬리 길이 20개 유지
                    if len(ball_trail) > 20:
                        ball_trail.pop(0)
            else:
                missing_frames += 1

            # ==========================================================
            # 🧠 [단순화된 판정 로직] STRIKE vs BALL
            # ==========================================================

            is_moving = False
            if len(ball_trail) > 3 and start_y != -1:
                if abs(ball_trail[-1][1] - start_y) > 30: is_moving = True

            if final_decision == "" and is_moving:

                curr_p = ball_trail[-1]
                prev_p = ball_trail[-2]

                # 1. [STRIKE] 박스 통과
                if check_box_hit(prev_p, curr_p, ABS_ZONE):
                    final_decision = "STRIKE"
                    print("  >>> ⚾ STRIKE 확정!")

                # 2. [BALL] 박스 안 통과하고 바닥으로 떨어짐
                elif curr_p[1] > zone_y2:
                    final_decision = "BALL"
                    print("  >>> BALL 확정 (바닥 통과)")

                # 3. [BALL] 화면 이탈 (폭투)
                elif curr_p[1] < 0 or curr_p[1] > h or curr_p[0] < 0 or curr_p[0] > w:
                    final_decision = "BALL"
                    print("  >>> BALL 확정 (화면 이탈)")

            # 4. [BALL] 놓침 감지 (공 사라짐)
            if final_decision == "" and is_moving:
                if missing_frames > 3:
                    final_decision = "BALL"
                    print("  >>> BALL 확정 (공 놓침)")

            # ==========================================================
            # 🎨 [디자인] 동글동글 혜성 꼬리
            # ==========================================================

            trail_color = (255, 255, 255)  # 기본: 흰색

            if final_decision == "STRIKE":
                trail_color = (255, 0, 0)  # 파랑
            elif final_decision == "BALL":
                trail_color = (0, 255, 0)  # 초록

            # [수정] 선(line) 안 긋고 '원(circle)'만 그려서 동글동글하게 표현
            for j in range(len(ball_trail)):
                # 꼬리 앞쪽(최신)은 크고, 뒤쪽(과거)은 작게
                # radius: 2px ~ 10px
                radius = int(2 + (j / len(ball_trail)) * 8)

                # 겹쳐서 그리면 자연스럽게 이어짐
                cv2.circle(annotated_frame, ball_trail[j], radius, trail_color, -1, cv2.LINE_AA)

            # 텍스트
            if final_decision:
                text_color = trail_color
                cv2.putText(annotated_frame, final_decision, (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, text_color, 7, cv2.LINE_AA)

            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f"  -> 결과 저장: {save_name}")

    cv2.destroyAllWindows()
    print("완료")