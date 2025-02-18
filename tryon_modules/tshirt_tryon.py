import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the t-shirt image with alpha (transparent background)
tshirt_img = cv2.imread('tshirt.png', cv2.IMREAD_UNCHANGED)
# Convert BGR to RGB
tshirt_img = cv2.cvtColor(tshirt_img, cv2.COLOR_BGRA2RGBA)

# Open the video capture (camera)
cap = cv2.VideoCapture(0)

# Adjustable parameters for t-shirt positioning
HORIZONTAL_ADJUST = 0  # Positive values move right, negative values move left
VERTICAL_ADJUST = 2    # Positive values move down, negative values move up
SCALE_FACTOR = 1.0    # Values > 1 increase size, values < 1 decrease size

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay `img_overlay` on top of `img` at the position specified by `pos` and blend using `alpha_mask`."""
    x, y = pos
    
    # Image and overlay sizes
    h, w = img_overlay.shape[:2]
    
    # Check if overlay position is within the image boundaries
    if y >= img.shape[0] or x >= img.shape[1] or y + h < 0 or x + w < 0:
        return  # Overlay is out of bounds
    
    # Crop overlay and mask if necessary
    if x < 0:
        img_overlay = img_overlay[:, -x:]
        alpha_mask = alpha_mask[:, -x:]
        x = 0
    if y < 0:
        img_overlay = img_overlay[-y:, :]
        alpha_mask = alpha_mask[-y:, :]
        y = 0
    if y + h > img.shape[0]:
        img_overlay = img_overlay[:img.shape[0]-y, :]
        alpha_mask = alpha_mask[:img.shape[0]-y, :]
    if x + w > img.shape[1]:
        img_overlay = img_overlay[:, :img.shape[1]-x]
        alpha_mask = alpha_mask[:, :img.shape[1]-x]
    
    # Blend images
    alpha_expanded = np.expand_dims(alpha_mask, axis=2)
    img_slice = img[y:y+img_overlay.shape[0], x:x+img_overlay.shape[1]]
    img[y:y+img_overlay.shape[0], x:x+img_overlay.shape[1]] = (
        img_slice * (1 - alpha_expanded) + img_overlay * alpha_expanded
    ).astype(np.uint8)

# Main loop for live video and processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        # Get the coordinates of the shoulders and hips
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))

        # Calculate the width and height of the t-shirt based on the shoulder and hip positions
        tshirt_width = int(np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder)) * 1.2 * SCALE_FACTOR)
        tshirt_height = int(np.linalg.norm(np.array(left_shoulder) - np.array(left_hip)) * 1.5 * SCALE_FACTOR)

        # Resize the t-shirt image
        resized_tshirt = cv2.resize(tshirt_img, (tshirt_width, tshirt_height), interpolation=cv2.INTER_AREA)

        # Determine position to place the t-shirt (centered between shoulders)
        x_offset = (left_shoulder[0] + right_shoulder[0]) // 2 - resized_tshirt.shape[1] // 2 + HORIZONTAL_ADJUST
        y_offset = left_shoulder[1] - int(tshirt_height * 0.2) + VERTICAL_ADJUST  # Align with the shoulder height

        # Get the alpha channel from the PNG (transparency mask)
        if resized_tshirt.shape[2] == 4:  # Check if the image has an alpha channel
            alpha_tshirt = resized_tshirt[:, :, 3] / 255.0
            overlay_img = cv2.cvtColor(resized_tshirt[:, :, :3], cv2.COLOR_RGB2BGR)
            overlay_image_alpha(frame, overlay_img, (x_offset, y_offset), alpha_tshirt)

    # Display the result
    cv2.imshow('Virtual T-Shirt Try-On', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        VERTICAL_ADJUST -= 5
    elif key == ord('s'):
        VERTICAL_ADJUST += 5
    elif key == ord('a'):
        HORIZONTAL_ADJUST -= 5
    elif key == ord('d'):
        HORIZONTAL_ADJUST += 5
    elif key == ord('+'):
        SCALE_FACTOR += 0.1
    elif key == ord('-'):
        SCALE_FACTOR -= 0.1

# Release resources
cap.release()
cv2.destroyAllWindows()