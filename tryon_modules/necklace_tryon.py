import cv2
import mediapipe as mp
import numpy as np
import sys

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def necklace_tryon(necklace_image_path):
    # Load the necklace image with alpha (transparent background)
    necklace_img = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)

    # Check if the necklace image was loaded successfully
    if necklace_img is None:
        print(f"Error: Could not load the necklace image. Please check the file path: {necklace_image_path}")
        return
    else:
        print(f"Necklace image loaded successfully with shape: {necklace_img.shape}")

    # Open the video capture (camera)
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Face Mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
        x, y = pos

        # Image and overlay sizes
        h, w = img_overlay.shape[:2]

        # Calculate the valid region for overlay
        y1, y2 = max(0, y), min(img.shape[0], y + h)
        x1, x2 = max(0, x), min(img.shape[1], x + w)

        # Adjust overlay region if necessary
        overlay_y1 = 0 if y >= 0 else -y
        overlay_y2 = h if y + h <= img.shape[0] else img.shape[0] - y
        overlay_x1 = 0 if x >= 0 else -x
        overlay_x2 = w if x + w <= img.shape[1] else img.shape[1] - x

        # Ensure we're not trying to overlay outside the image bounds
        if y1 >= y2 or x1 >= x2 or overlay_y1 >= overlay_y2 or overlay_x1 >= overlay_x2:
            return

        # Extract the alpha channel from img_overlay
        if img_overlay.shape[2] == 4:
            alpha = img_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
            img_overlay = img_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]
        else:
            alpha = alpha_mask

        # Blend the overlay with the image
        if alpha is not None:
            for c in range(3):
                img[y1:y2, x1:x2, c] = img_overlay[:,:,c] * alpha + img[y1:y2, x1:x2, c] * (1 - alpha)
        else:
            img[y1:y2, x1:x2] = img_overlay

    # Main loop for live video and processing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Check if face landmarks are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the neck landmarks
                left_neck = face_landmarks.landmark[234]
                right_neck = face_landmarks.landmark[454]

                h, w, _ = frame.shape
                left_neck_coords = (int(left_neck.x * w), int(left_neck.y * h))
                right_neck_coords = (int(right_neck.x * w), int(right_neck.y * h))

                # Calculate the center of the neck
                neck_center = ((left_neck_coords[0] + right_neck_coords[0]) // 2,
                               (left_neck_coords[1] + right_neck_coords[1]) // 2)

                # Calculate the size for the necklace overlay
                necklace_width = int(abs(right_neck_coords[0] - left_neck_coords[0]) * 1.6)  # Adjust multiplier as needed
                necklace_height = int(necklace_img.shape[0] * (necklace_width / necklace_img.shape[1]))

                # Resize the necklace image
                resized_necklace = cv2.resize(necklace_img, (necklace_width, necklace_height), interpolation=cv2.INTER_AREA)

                # Determine position to place the necklace
                x_offset = neck_center[0] - resized_necklace.shape[1] // 2
                y_offset = neck_center[1] + resized_necklace.shape[0] // 3  # Adjusted to move necklace lower

                # Overlay necklace on the frame using alpha mask
                overlay_image_alpha(frame, resized_necklace, (x_offset, y_offset), None)

        # Display the result
        cv2.imshow('Virtual Necklace Try-On', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        necklace_image_path = sys.argv[1]
        necklace_tryon(necklace_image_path)
    else:
        print("Please provide the path to the necklace image.")