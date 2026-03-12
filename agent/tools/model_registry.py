"""Singleton lazy model loader for Grounding DINO and CLIP."""
import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    CLIPModel,
    CLIPProcessor,
)

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._gdino_processor = None
            cls._instance._gdino_model = None
            cls._instance._clip_model = None
            cls._instance._clip_processor = None
            cls._instance._device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._instance

    @property
    def device(self) -> str:
        return self._device

    def load_gdino(self):
        if self._gdino_model is None:
            self._gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
            self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GDINO_MODEL_ID
            ).to(self._device)
            self._gdino_model.eval()

    def load_clip(self):
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(
                self._device
            )
            self._clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
            self._clip_model.eval()

    def ensure_all(self):
        self.load_gdino()
        self.load_clip()

    @property
    def gdino_processor(self):
        self.load_gdino()
        return self._gdino_processor

    @property
    def gdino_model(self):
        self.load_gdino()
        return self._gdino_model

    @property
    def clip_model(self):
        self.load_clip()
        return self._clip_model

    @property
    def clip_processor(self):
        self.load_clip()
        return self._clip_processor


registry = ModelRegistry()
