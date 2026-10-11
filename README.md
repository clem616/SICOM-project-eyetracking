# Eye-gaze speller (first prototype)

Write letters by looking in a direction: a webcam follows the eyes, and two successive gaze directions select one letter. No special hardware is needed.

This is the first prototype, written in spring 2025 as my second-year project at Grenoble INP – Phelma (SICOM track). It is inspired by the paper communication sheets used by people who can only move their eyes, where a first glance picks a group of letters and a second glance picks the letter. The code is kept here as it was; a cleaned-up version of the later, working speller is in progress.

## How it works

1. **Find the eyes.** OpenCV reads the webcam; MediaPipe Face Mesh gives the face landmarks, from which each eye region is cropped.
2. **Isolate the iris.** The eye image is thresholded (Otsu) and cleaned with morphological operations (scikit-image) to keep the dark iris and pupil.
3. **Measure where it is.** The barycentre of the dark pixels, relative to the eye region, tells where the person is looking.
4. **Decide the direction.** The position is classified into one of six zones (top-left, top-middle, top-right, bottom-left, bottom-middle, bottom-right), with a majority vote over several frames to absorb blinks and noise.
5. **Select the letter.** Two successive directions index a letter on the digital version of the communication sheet.

A calibration phase records the user's eyes for each direction before use.

## Files

| File | Role |
|---|---|
| `speller_eyetracking_image_treatment_only.py` | The speller using image processing only: calibration, direction detection, letter selection. |
| `Barycentre.py` | Feature extraction from eye images: barycentre, eye opening, export to CSV for analysis. |
| `speller_eyetracking_NN_test_direct.py` | Experiment: a small neural network (Keras) classifying the gaze direction from the same features. |
| `blue_reflects_treatments_tests.py` | Experiments on removing the blue screen reflections that glasses add to the eye image. |
| `Interface_user_souris.py` | The letter-selection interface driven by the mouse, to test it without the camera. |

## Running it

```bash
pip install opencv-python mediapipe numpy scikit-image pillow
python3 speller_eyetracking_image_treatment_only.py
```

The scripts contain paths from my own machine for the calibration images; change `root_output_dir` at the top of the file before running.

## State of the project

- Comments and names are in French, and the scripts are long single files: this is prototype code.
- The eye images used for calibration and tests are not included, as they are personal images.
- Known weak points at this stage: sensitivity to lighting and to reflections on glasses, and thresholds that depend on the user and on the position of the head.
