import cv2
import numpy as np
import face_recognition
import pickle
import sys
import tempfile
from pathlib import Path

from virtual_tryon.glasses_fitting import (
    fit_glasses_overlay,
    create_edit_mask
)
from virtual_tryon.ai_photo_edit import generate_photorealistic_tryon


def process_glasses_image(
    photo_path,
    glasses_path,
    run_ai=True,
    output_dir="output"
):
    """
    Complete hybrid glasses try-on pipeline. The fast preview trims temple arms
    from transparent glasses assets; AI editing keeps the original reference.

    Args:
        photo_path: Path to the person's photograph.
        glasses_path: Path to the glasses image/reference.
        run_ai: Whether to generate the photorealistic AI result.
        output_dir: Directory for generated images.

    Returns:
        Dictionary containing paths to generated results.
    """

    photo_path = Path(photo_path)
    glasses_path = Path(glasses_path)
    output_dir = Path(output_dir)

    if not photo_path.is_file():
        raise FileNotFoundError(f"Photo not found: {photo_path}")

    if not glasses_path.is_file():
        raise FileNotFoundError(f"Glasses image not found: {glasses_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "preview": None,
        "mask": None,
        "final": None
    }

    # STEP 1: Fast glasses fitting preview
    print("[1/3] Fitting glasses to face...")

    try:
        results["preview"] = _fit_glasses_preview(
            str(photo_path),
            str(glasses_path),
            str(output_dir / "fitted_preview.png"),
        )

        print(f"Preview saved: {results['preview']}")

    except ValueError as error:
        # Normal product photos may not have transparent backgrounds.
        print(f"Skipping overlay preview: {error}")

    # STEP 2: Generate face-aware editing mask
    print("[2/3] Detecting face and creating editing mask...")

    results["mask"] = create_edit_mask(
        str(photo_path),
        output_path=str(output_dir / "glasses_mask.png")
    )

    print(f"Mask saved: {results['mask']}")

    # STEP 3: Photorealistic AI editing
    if run_ai:
        print("[3/3] Generating photorealistic glasses try-on...")

        results["final"] = generate_photorealistic_tryon(
            photo_path=str(photo_path),
            glasses_reference_path=str(glasses_path),
            output_path=str(output_dir / "ai_tryon.png"),
            mask_path=results["mask"]
        )

        print(f"AI result saved: {results['final']}")

    else:
        print("[3/3] AI editing skipped.")

    return results

def _find_glasses_eye_openings(glasses_image):
    alpha_mask = (glasses_image[:, :, 3] >= 128).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        alpha_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return None

    image_area = glasses_image.shape[0] * glasses_image.shape[1]
    openings = []
    for index, contour in enumerate(contours):
        if hierarchy[0][index][3] < 0:
            continue
        area = abs(cv2.contourArea(contour))
        if area < image_area * 0.02:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
        openings.append((area, center, cv2.boundingRect(contour)))

    if len(openings) < 2:
        return None

    openings.sort(key=lambda opening: opening[0], reverse=True)
    first, second = sorted(openings[:2], key=lambda opening: opening[1][0])
    if min(first[0], second[0]) / max(first[0], second[0]) < 0.5:
        return None

    left_center, right_center = first[1], second[1]
    image_height, image_width = glasses_image.shape[:2]
    if (
        right_center[0] - left_center[0] < image_width * 0.2
        or abs(right_center[1] - left_center[1]) > image_height * 0.2
    ):
        return None
    return first, second


def _find_glasses_eye_centers(glasses_image):
    openings = _find_glasses_eye_openings(glasses_image)
    if openings is None:
        return None
    return openings[0][1], openings[1][1]


def _remove_glasses_temples(glasses_image):
    if (
        glasses_image is None
        or glasses_image.ndim != 3
        or glasses_image.shape[2] != 4
    ):
        return glasses_image

    openings = _find_glasses_eye_openings(glasses_image)
    if openings is None:
        return glasses_image

    image_height = glasses_image.shape[0]
    top = min(opening[2][1] for opening in openings)
    bottom = max(opening[2][1] + opening[2][3] for opening in openings)
    padding = int(max(opening[2][3] for opening in openings) * 0.2)
    top = max(0, top - padding)
    bottom = min(image_height, bottom + padding)
    if top == 0 and bottom == image_height:
        return glasses_image
    return glasses_image[top:bottom, :]


def _fit_glasses_preview(photo_path, glasses_path, output_path):
    glasses_image = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
    if glasses_image is None:
        return fit_glasses_overlay(
            photo_path, glasses_path, output_path=output_path
        )

    front_frame = _remove_glasses_temples(glasses_image)
    if front_frame is glasses_image:
        return fit_glasses_overlay(
            photo_path, glasses_path, output_path=output_path
        )

    with tempfile.TemporaryDirectory(
        prefix="glasses-preview-", dir=str(Path(output_path).parent)
    ) as temp_dir:
        front_frame_path = Path(temp_dir) / "front_frame.png"
        if not cv2.imwrite(str(front_frame_path), front_frame):
            raise OSError(f"Could not write cropped glasses reference: {front_frame_path}")
        return fit_glasses_overlay(
            photo_path, str(front_frame_path), output_path=output_path
        )


def _apply_glasses_to_faces(frame, glasses_image, faces):
    result = frame.copy()
    frame_h, frame_w = result.shape[:2]
    source_h, source_w = glasses_image.shape[:2]
    source_eye_centers = _find_glasses_eye_centers(glasses_image)
    if source_eye_centers is None:
        source_eye_distance = source_w / 2.8
        source_midpoint = (source_w / 2.0, source_h / 2.0)
        source_eye_centers = (
            (source_midpoint[0] - source_eye_distance / 2.0, source_midpoint[1]),
            (source_midpoint[0] + source_eye_distance / 2.0, source_midpoint[1]),
        )

    source_left, source_right = source_eye_centers
    source_distance = float(np.hypot(
        source_right[0] - source_left[0],
        source_right[1] - source_left[1],
    ))
    if source_distance < 2:
        return result

    source_midpoint = (
        (source_left[0] + source_right[0]) / 2.0,
        (source_left[1] + source_right[1]) / 2.0,
    )
    source_angle = np.arctan2(
        source_right[1] - source_left[1],
        source_right[0] - source_left[0],
    )

    for landmarks in faces:
        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")
        if not left_eye or not right_eye:
            continue

        eye_centers = sorted(
            (np.mean(left_eye, axis=0), np.mean(right_eye, axis=0)),
            key=lambda center: center[0],
        )
        target_left, target_right = eye_centers
        eye_distance = float(np.hypot(
            target_right[0] - target_left[0],
            target_right[1] - target_left[1],
        ))
        if eye_distance < 2:
            continue

        target_midpoint = (
            (target_left[0] + target_right[0]) / 2.0,
            (target_left[1] + target_right[1]) / 2.0,
        )
        target_angle = np.arctan2(
            target_right[1] - target_left[1],
            target_right[0] - target_left[0],
        )
        scale = eye_distance / source_distance
        rotation = np.degrees(source_angle - target_angle)
        matrix = cv2.getRotationMatrix2D(source_midpoint, rotation, scale)
        matrix[0, 2] += target_midpoint[0] - source_midpoint[0]
        matrix[1, 2] += target_midpoint[1] - source_midpoint[1]

        corners = np.array(
            [[[0, 0], [source_w, 0], [source_w, source_h], [0, source_h]]],
            dtype=np.float32,
        )
        transformed_corners = cv2.transform(corners, matrix)[0]
        x1 = int(np.floor(transformed_corners[:, 0].min()))
        y1 = int(np.floor(transformed_corners[:, 1].min()))
        x2 = int(np.ceil(transformed_corners[:, 0].max()))
        y2 = int(np.ceil(transformed_corners[:, 1].max()))

        dx1, dy1 = max(0, x1), max(0, y1)
        dx2, dy2 = min(frame_w, x2), min(frame_h, y2)
        if dx1 >= dx2 or dy1 >= dy2:
            continue

        matrix[0, 2] -= x1
        matrix[1, 2] -= y1
        overlay = cv2.warpAffine(
            glasses_image, matrix, (x2 - x1, y2 - y1),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        overlay = overlay[dy1 - y1:dy2 - y1, dx1 - x1:dx2 - x1]
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        foreground = overlay[:, :, :3].astype(np.float32)
        background = result[dy1:dy2, dx1:dx2].astype(np.float32)
        result[dy1:dy2, dx1:dx2] = (
            foreground * alpha + background * (1.0 - alpha)
        ).astype(np.uint8)

    return result


def _validate_glasses_inputs(frame, glasses_image):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("The photo must be a non-empty BGR image.")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("The photo must be a non-empty BGR image.")
    if glasses_image is None or glasses_image.ndim != 3 or glasses_image.shape[2] != 4:
        raise ValueError("The glasses asset must be a transparent PNG (RGBA).")
    if glasses_image.shape[0] == 0 or glasses_image.shape[1] == 0:
        raise ValueError("The glasses image must not be empty.")


def apply_glasses_filter(frame, glasses_image):
    """Overlay a transparent glasses PNG on detected faces in a BGR frame."""
    if frame is None or glasses_image is None:
        return frame

    _validate_glasses_inputs(frame, glasses_image)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_landmarks(rgb_frame)
    glasses_image = _remove_glasses_temples(glasses_image)
    return _apply_glasses_to_faces(frame, glasses_image, faces)


def process_glasses_frame(frame, glasses_image):
    """Return a glasses preview and detected face count for a BGR frame."""
    _validate_glasses_inputs(frame, glasses_image)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_landmarks(rgb_frame)
    glasses_image = _remove_glasses_temples(glasses_image)
    return _apply_glasses_to_faces(frame, glasses_image, faces), len(faces)


def load_face_predictor(model_dir="trainer"):
    """
    Load the LBPH model and ID-to-name dictionary created by train.py.
    Returns (recognizer, names_dict), or (None, {}) if not trained yet.
    """
    model_dir = Path(model_dir)
    model_dirs = [model_dir] if model_dir.is_absolute() else [Path.cwd() / model_dir]
    if not model_dir.is_absolute() and getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent / model_dir
        if executable_dir not in model_dirs:
            model_dirs.append(executable_dir)

    model_dir = next(
        (
            candidate for candidate in model_dirs
            if (candidate / "trainer.yml").is_file()
            and (candidate / "names.pkl").is_file()
        ),
        None,
    )
    if model_dir is None:
        print("[INFO] Face model not found. Run train.py first.")
        return None, {}

    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        print("[ERROR] LBPH requires opencv-contrib-python.")
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_dir / "trainer.yml"))
    with (model_dir / "names.pkl").open("rb") as file:
        names_dict = pickle.load(file)

    return recognizer, names_dict


def predict_faces(frame, recognizer, names_dict, confidence_threshold=70):
    """
    Predict identities in a BGR webcam frame using the project's LBPH model.
    Returns an annotated copy. Lower LBPH distance means a better match;
    predictions above confidence_threshold are labeled Unknown.
    """
    if frame is None or recognizer is None:
        return frame.copy() if frame is not None else frame

    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        label_id, distance = recognizer.predict(face_roi)
        person_name = names_dict.get(label_id, "Unknown")

        if distance > confidence_threshold:
            display_name = "Unknown"
        else:
            display_name = str(person_name)

        # Draw the identity box on the preview frame.
        color = (0, 200, 0) if display_name != "Unknown" else (0, 165, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            output, f"{display_name} ({distance:.1f})",
            (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color, 2
        )

    return output


def process_webcam_frame(
    frame,
    glasses_image=None,
    recognizer=None,
    names_dict=None,
    confidence_threshold=70,
):
    """Apply the glasses overlay and annotate recognized people in a webcam frame."""
    if frame is None:
        return None

    display_frame = (
        apply_glasses_filter(frame, glasses_image)
        if glasses_image is not None
        else frame.copy()
    )
    if recognizer is not None:
        display_frame = predict_faces(
            display_frame,
            recognizer,
            names_dict or {},
            confidence_threshold,
        )
    return display_frame



def run_glasses_preview():
    print("\n--- Hybrid Smart Glasses Try-On ---")

    photo = input("Person's photo path: ").strip().strip('"')
    glasses = input("Glasses image path: ").strip().strip('"')

    try:
        results = process_glasses_image(
            photo_path=photo,
            glasses_path=glasses,
            run_ai=True
        )

        print("\n--- Processing Complete ---")

        for stage, path in results.items():
            print(f"{stage}: {path}")

    except Exception as error:
        print(f"[ERROR] {error}")
