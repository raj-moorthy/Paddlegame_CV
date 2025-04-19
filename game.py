import cv2
import mediapipe as mp
import time
import json
import numpy as np
import os

SCORE_FILE = "high_scores.json"

def load_scores():
    if os.path.exists(SCORE_FILE):
        with open(SCORE_FILE, "r") as f:
            return json.load(f)
    return []

def save_score(score):
    scores = load_scores()
    scores.append(score)
    scores = sorted(scores, reverse=True)[:5]
    with open(SCORE_FILE, "w") as f:
        json.dump(scores, f)

def show_webcam_hand(frame, results, mp_hands):
    annotated_frame = frame.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return annotated_frame

def play_game():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    WIDTH, HEIGHT = 640, 480
    paddle_width, paddle_height = 100, 15
    paddle_y = HEIGHT - 40
    paddle_x = WIDTH // 2 - paddle_width // 2
    ball_pos = [WIDTH // 2, HEIGHT // 2]
    ball_vel = [7, 7]
    score = 0
    game_started = False
    game_over = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_game = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            finger_x = int(lm[8].x * WIDTH)
            paddle_x = max(0, min(WIDTH - paddle_width, finger_x - paddle_width // 2))

            fingers_up = [
                lm[8].y < lm[6].y,
                lm[12].y < lm[10].y,
                lm[16].y < lm[14].y,
                lm[20].y < lm[18].y
            ]
            if all(fingers_up) and not game_started:
                game_started = True

        cv2.rectangle(frame_game, (paddle_x, paddle_y),
                      (paddle_x + paddle_width, paddle_y + paddle_height), (0, 255, 0), -1)

        if game_started:
            ball_pos[0] += ball_vel[0]
            ball_pos[1] += ball_vel[1]

            if ball_pos[0] <= 0 or ball_pos[0] >= WIDTH:
                ball_vel[0] *= -1
            if ball_pos[1] <= 0:
                ball_vel[1] *= -1

            if (paddle_y <= ball_pos[1] + 10 <= paddle_y + paddle_height) and \
               (paddle_x <= ball_pos[0] <= paddle_x + paddle_width):
                ball_vel[1] *= -1
                score += 1

            if ball_pos[1] > HEIGHT:
                game_over = True
                save_score(score)

        cv2.circle(frame_game, tuple(ball_pos), 10, (0, 0, 255), -1)
        cv2.putText(frame_game, f"Score: {score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if not game_started:
            cv2.putText(frame_game, "Show Open Hand to Start", (120, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if game_over:
            cv2.putText(frame_game, "GAME OVER", (WIDTH // 2 - 100, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.imshow("Gesture Game", frame_game)
            cv2.waitKey(2000)
            break

        # Show game window
        cv2.imshow("Gesture Game", frame_game)

        # Show webcam with hand landmarks
        webcam_display = show_webcam_hand(frame, results, mp_hands)
        cv2.imshow("Hand Detection", webcam_display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Game Over! Your score: {score}")

if __name__ == "__main__":
    play_game()
