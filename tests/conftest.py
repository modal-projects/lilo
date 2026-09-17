"""Keep image-definition imports offline during CPU tests."""
import os

# No image is built by this suite. Resolver tests explicitly clear this override.
os.environ.setdefault("LILO_MILES_COMMIT", "a" * 40)
