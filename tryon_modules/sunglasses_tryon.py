import cv2
import mediapipe as mp
import numpy as np
import sys





# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

def try_sunglasses(sunglasses_path):
    # Load the selected sunglasses image
    sunglasses_img = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)

    # Check if the sunglasses image was loaded successfully
    if sunglasses_img is None:
        print(f"Error: Unable to load image at {sunglasses_path}. Please check the file path.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame from webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]

                h, w, _ = frame.shape
                left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
                right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

                # Calculate the distance between the eyes
                eye_distance = np.linalg.norm(np.array(left_eye_coords) - np.array(right_eye_coords))
                scale_factor = 1.8
                sunglasses_width = int(eye_distance * scale_factor)
                sunglasses_height = int(sunglasses_width * sunglasses_img.shape[0] / sunglasses_img.shape[1])

                if sunglasses_width > 0 and sunglasses_height > 0:
                    resized_sunglasses = cv2.resize(sunglasses_img, (sunglasses_width, sunglasses_height))

                    x_offset = (left_eye_coords[0] + right_eye_coords[0]) // 2 - sunglasses_width // 2
                    y_offset = left_eye_coords[1] - sunglasses_height // 2

                    if 0 <= x_offset < w - sunglasses_width and 0 <= y_offset < h - sunglasses_height:
                        # Check if the sunglasses image has an alpha channel
                        if resized_sunglasses.shape[2] == 4:
                            alpha_sunglasses = resized_sunglasses[:, :, 3] / 255.0
                        else:
                            alpha_sunglasses = np.ones((resized_sunglasses.shape[0], resized_sunglasses.shape[1]), dtype=np.float32)  # Fully opaque

                        # Blend the sunglasses with the frame
                        for c in range(3):  # RGB channels
                            frame[y_offset:y_offset+sunglasses_height, x_offset:x_offset+sunglasses_width, c] = \
                                (alpha_sunglasses * resized_sunglasses[:, :, c] +
                                 (1 - alpha_sunglasses) * frame[y_offset:y_offset+sunglasses_height, x_offset:x_offset+sunglasses_width, c])

        cv2.imshow("Sunglasses Try-On", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Get the selected sunglasses image path from the command-line argument
    if len(sys.argv) > 1:
        sunglasses_image_path = sys.argv[1]
    else:
        sunglasses_image_path = 'static/sunglasses1.png'  # Default to first image
    try_sunglasses(sunglasses_image_path)