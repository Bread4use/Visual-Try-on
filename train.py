import os
import pickle
from pathlib import Path

import cv2
import numpy as np


def load_dataset_from_folders(data_dir="dataset", progress_callback=None):
    face_samples = []
    ids = []
    names_dict = {}

    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        return face_samples, np.array(ids), names_dict

    person_folders = sorted(p for p in dataset_path.iterdir() if p.is_dir())
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_detector.empty():
        raise RuntimeError("OpenCV's Haar face detector could not be loaded.")

    for current_id, person_folder in enumerate(person_folders):
        person_name = person_folder.name
        names_dict[current_id] = person_name
        image_paths = [
            p for p in person_folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

        if progress_callback:
            progress_callback(
                f"Scanning {person_name} ({len(image_paths)} image(s))..."
            )

        for image_path in image_paths:
            gray_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                continue

            faces = face_detector.detectMultiScale(
                gray_img, scaleFactor=1.2, minNeighbors=5
            )
            for (x, y, w, h) in faces:
                face_samples.append(gray_img[y:y + h, x:x + w])
                ids.append(current_id)

    return face_samples, np.array(ids, dtype=np.int32), names_dict


def train_lbph_model(data_dir="dataset", output_dir="trainer",
                     progress_callback=None):
    """Train and save an LBPH model. Raises clear errors if training cannot run."""
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face is unavailable. Install opencv-contrib-python "
            "(and uninstall conflicting opencv-python if necessary)."
        )

    if progress_callback:
        progress_callback("Scanning dataset and detecting faces...")

    faces, ids, names_dict = load_dataset_from_folders(
        data_dir, progress_callback
    )

    if len(faces) == 0:
        raise ValueError(
            f"No faces were detected in '{data_dir}'. Add clear, front-facing "
            "photos to dataset/person_name/ and try again."
        )

    if progress_callback:
        progress_callback(f"Training on {len(faces)} face sample(s)...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, ids)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(output_path / "trainer.yml"))

    with (output_path / "names.pkl").open("wb") as file:
        pickle.dump(names_dict, file)

    result = {
        "samples": len(faces),
        "people": len(names_dict),
        "model": str(output_path / "trainer.yml"),
        "names": str(output_path / "names.pkl"),
    }
    if progress_callback:
        progress_callback(
            f"Finished: {result['samples']} face sample(s), "
            f"{result['people']} person/people."
        )
    return result
