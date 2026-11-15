SMART GLASSES STUDIO

RUN
  python __main__.py

This opens the desktop GUI for photo upload, webcam capture, LBPH training,
live identity predictions when a trained model is available, and glasses
preview.

The webcam page loads identity predictions from trainer/trainer.yml and
trainer/names.pkl. For the packaged executable, model lookup checks both the
current working directory and the folder beside the executable. Train a model
from the GUI if neither location contains a trained model.

DATASET LAYOUT
Photos are stored directly under one folder per person:

dataset/
  alice/
    photo_1.jpg
    photo_2.jpg
  bob/
    photo_1.jpg

The upload and webcam capture tools use this layout. Names are stored in
lowercase. Avoid path separators in person names.

TRAINING
The current train.py scans direct person folders under dataset/ and trains
LBPH on every face it detects. It does not create train/test splits, validate
recording sessions, check for duplicates, or report evaluation accuracy.
Do not treat its training results as an independent accuracy evaluation.
For a meaningful evaluation, keep separate recording sessions and implement
split-aware training and testing before making accuracy claims.

GLASSES PREVIEW
The GUI's fast preview overlays a transparent RGBA glasses PNG on faces found
by face_recognition. Choose a face photo and a transparent glasses PNG before
clicking Preview. The live webcam filter in take_photo.py also uses this
overlay; captured training photos remain unfiltered.

The optional hybrid AI try-on pipeline in glasses_filter.py can also create a
fitted preview, an editing mask, and an AI-generated result. The AI step
requires the openai package and OPENAI_API_KEY. The GUI preview does not call
the AI service.

DEPENDENCIES
Install the packages required by the project, commonly:
  pip install opencv-contrib-python numpy face-recognition pillow

OpenCV's contrib build is needed for cv2.face.LBPHFaceRecognizer_create().
Use one OpenCV package consistently; avoid conflicting opencv-python and
opencv-contrib-python installations.
