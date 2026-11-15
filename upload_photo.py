import cv2
from pathlib import Path


def upload_photo(person_name, photo_path, data_dir="dataset"):
    """Save a selected image in dataset/<person_name>/ and return its path."""
    person_name = (person_name or "").strip().lower()
    if not person_name:
        raise ValueError("Please enter a person's name.")
    if person_name in {".", ".."} or any(
        character in person_name for character in ("/", "\\")
    ):
        raise ValueError("Person name must not contain path separators.")

    source = Path(photo_path)
    if not source.is_file():
        raise FileNotFoundError(f"Photo not found: {source}")

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError("OpenCV could not read this image. Try a JPG or PNG file.")

    person_folder = Path(data_dir) / person_name
    person_folder.mkdir(parents=True, exist_ok=True)

    # Find the next unused number rather than relying on the number of files.
    number = 1
    while (person_folder / f"photo_{number}.jpg").exists():
        number += 1

    destination = person_folder / f"photo_{number}.jpg"
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not save photo to {destination}")

    return destination
