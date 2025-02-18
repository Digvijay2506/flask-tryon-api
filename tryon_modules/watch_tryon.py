import cv2
import mediapipe as mp
import numpy as np
import sys

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    h, w = img_overlay.shape[:2]
    y1, y2 = max(0, y), min(img.shape[0], y + h)
    x1, x2 = max(0, x), min(img.shape[1], x + w)
    overlay_y1 = 0 if y >= 0 else -y
    overlay_y2 = h if y + h <= img.shape[0] else img.shape[0] - y
    overlay_x1 = 0 if x >= 0 else -x
    overlay_x2 = w if x + w <= img.shape[1] else img.shape[1] - x
    if y1 >= y2 or x1 >= x2 or overlay_y1 >= overlay_y2 or overlay_x1 >= overlay_x2:
        return
    if img_overlay.shape[2] == 4:
        alpha = img_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
        img_overlay = img_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]
    else:
        alpha = None
    if alpha is not None:
        for c in range(3):
            img[y1:y2, x1:x2, c] = img_overlay[:,:,c] * alpha + img[y1:y2, x1:x2, c] * (1 - alpha)
    else:
        img[y1:y2, x1:x2] = img_overlay

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def watch_tryon(watch_image_path):
    watch_img = cv2.imread(watch_image_path, cv2.IMREAD_UNCHANGED)
    if watch_img is None:
        print(f"Error: Could not load the watch image. Please check the file path and name: {watch_image_path}")
        return
    else:
        print(f"Watch image loaded successfully with shape: {watch_img.shape}")

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                h, w, _ = frame.shape
                wrist_coords = (int(wrist.x * w), int(wrist.y * h))
                index_mcp_coords = (int(index_mcp.x * w), int(index_mcp.y * h))
                dx = index_mcp_coords[0] - wrist_coords[0]
                dy = index_mcp_coords[1] - wrist_coords[1]
                angle = np.degrees(np.arctan2(dy, dx)) - 90  # Adjust by 90 degrees
                watch_scale_factor = 0.10  # Adjust this value to make the watch smaller or larger
                watch_width = int(w * watch_scale_factor)
                watch_height = int(watch_img.shape[0] * (watch_width / watch_img.shape[1]))
                resized_watch = cv2.resize(watch_img, (watch_width, watch_height), interpolation=cv2.INTER_AREA)
                rotated_watch = rotate_image(resized_watch, angle)
                x_offset = wrist_coords[0] - rotated_watch.shape[1] // 2
                y_offset = wrist_coords[1] - rotated_watch.shape[0] // 2
                overlay_image_alpha(frame, rotated_watch, (x_offset, y_offset), None)

        cv2.imshow('Virtual Watch Try-On', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python watch_tryon.py <path_to_watch_image>")
        sys.exit(1)
    
    watch_image_path = sys.argv[1]
    watch_tryon(watch_image_path)
