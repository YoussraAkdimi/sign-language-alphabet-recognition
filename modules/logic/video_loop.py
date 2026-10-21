import cv2

import mediapipe as mp
from modules.dto.capture_context import CaptureContext
from modules.protocols.protocols import ImageAnnotator

def capture_video_data(capture_device, landmarker, image_annotator: ImageAnnotator, capture_context: CaptureContext):
    ret, frame = capture_device.read()
    if not ret:
        print("Failed to grab frame")
    else:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a MediaPipe image from the RGB frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect hand landmarks
        capture_context.detection_result = landmarker.detect_for_video(
            mp_image,
            capture_context.timestamp_ms
        )

        # annotate the frame with camera number and landmarks
        annotated_frame = image_annotator.annotate_image(frame, capture_context=capture_context)  # Annotate the frame with camera number

        # Display the annotated frame
        cv2.imshow('Video capture', annotated_frame)
