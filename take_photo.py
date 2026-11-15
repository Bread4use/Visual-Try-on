import cv2
from pathlib import Path
from glasses_filter import (
    load_face_predictor,
    process_webcam_frame,
)


def _person_directory(person_name, data_dir="dataset"):
    person_name = (person_name or "").strip().lower()
    if not person_name:
        raise ValueError("Please enter a person's name.")
    if person_name in {".", ".."} or any(
        character in person_name for character in ("/", "\\")
    ):
        raise ValueError("Person name must not contain path separators.")
    return Path(data_dir) / person_name


def _next_photo_path(person_dir):
    number = 1
    while (person_dir / f"photo_{number}.jpg").exists():
        number += 1
    return person_dir / f"photo_{number}.jpg"


def open_camera(camera_index=0):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def save_camera_frame(frame, person_name, data_dir="dataset"):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Camera frame must be a non-empty BGR image.")

    person_dir = _person_directory(person_name, data_dir)
    person_dir.mkdir(parents=True, exist_ok=True)
    destination = _next_photo_path(person_dir)
    if not cv2.imwrite(str(destination), frame):
        raise OSError(f"Could not save photo to {destination}")
    return destination


def take_webcam_photos():
    print("--- Webcam: Live Glasses Filter + Face Prediction ---")
    person_name = input("Who are you? Tell us your name! :) ").strip()
    if not person_name:
        print("Error: Name cannot be empty.")
        return

    try:
        save_dir = _person_directory(person_name)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    glasses_path_text = input(
        "Path to transparent glasses PNG [glasses.png]: "
    ).strip().strip('"')
    glasses_path = Path(glasses_path_text or "glasses.png")
    glasses_image = cv2.imread(str(glasses_path), cv2.IMREAD_UNCHANGED)

    filter_enabled = glasses_image is not None
    if filter_enabled and (glasses_image.ndim != 3 or glasses_image.shape[2] != 4):
        print("[WARNING] Glasses image must be a transparent RGBA PNG.")
        glasses_image = None
        filter_enabled = False
    elif not filter_enabled:
        print(f"[WARNING] Could not load '{glasses_path}'. Filter starts OFF.")

    recognizer, names_dict = load_face_predictor()
    if recognizer is None:
        print("[WARNING] Face prediction is OFF until you run option 3 to train the model.")

    try:
        cam = open_camera()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    img_counter = len(list(save_dir.glob("photo_*.jpg")))
    print("\nSPACE = save original photo | F = toggle glasses | Q = quit")
    print("Recognized names are displayed above detected faces.")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error: Failed to grab frame from camera.")
                break

            display_frame = frame.copy()
            try:
                display_frame = process_webcam_frame(
                    frame,
                    glasses_image if filter_enabled else None,
                    recognizer,
                    names_dict,
                )
            except Exception as exc:
                cv2.putText(
                    display_frame, f"Webcam processing error: {str(exc)[:45]}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1
                )

            status = "ON" if filter_enabled else "OFF"
            cv2.putText(
                display_frame,
                f"Saved: {img_counter} | Glasses: {status} | Prediction: "
                f"{'ON' if recognizer is not None else 'OFF'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
            )
            cv2.putText(
                display_frame, "SPACE: save | F: glasses on/off | Q: quit",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
            )

            cv2.imshow("Webcam Glasses + Face Prediction", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                try:
                    img_name = save_camera_frame(frame, person_name)
                    img_counter += 1
                    print(f"[SUCCESS] Saved original photo: {img_name}")
                except (OSError, ValueError) as exc:
                    print(f"[ERROR] Could not save photo: {exc}")

            elif key in (ord("f"), ord("F")):
                if glasses_image is not None:
                    filter_enabled = not filter_enabled
                else:
                    print("No valid transparent glasses PNG is loaded.")

            elif key in (ord("q"), ord("Q")):
                print("Quitting webcam...")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    print(f"\nFinished! Total photos saved: {img_counter}")


if __name__ == "__main__":
    take_webcam_photos()
