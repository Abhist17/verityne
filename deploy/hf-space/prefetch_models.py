"""Pull the checkpoints that load at boot, at image build time.

A free Space sleeps after inactivity and cold-starts on the next request. If
these two are not already in the image, that request pays a ~600 MB download
before it sees a byte of the API.

The classifier's id is read from the same marker file the application reads, so
this cannot warm one checkpoint while the app loads a different one.
"""
from pathlib import Path

from facenet_pytorch import InceptionResnetV1
from transformers import AutoImageProcessor, AutoModelForImageClassification

FALLBACK = "prithivMLmods/Deep-Fake-Detector-v2-Model"
marker = Path("/app/storage/models/active_detector.txt")
repo = marker.read_text().strip() if marker.exists() else FALLBACK

print(f"pre-pulling classifier: {repo}", flush=True)
AutoImageProcessor.from_pretrained(repo)
AutoModelForImageClassification.from_pretrained(repo)

print("pre-pulling face embedder: facenet vggface2", flush=True)
InceptionResnetV1(pretrained="vggface2").eval()

print("checkpoints cached into the image", flush=True)
