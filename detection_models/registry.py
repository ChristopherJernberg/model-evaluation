import re
from re import Pattern
from typing import Callable

from detection_models.base_models import Detector

# Registry format: (pattern, factory_function, is_pose_capable)
ModelRegistryEntry = tuple[Pattern, Callable[[str, str, float, float], Detector], bool]


class ModelRegistry:
  _registry: list[ModelRegistryEntry] = []

  @classmethod
  def register(cls, pattern: str, is_pose_capable: bool = False):
    """Decorator to register a model factory function"""

    def decorator(factory_func):
      compiled_pattern = re.compile(pattern)
      cls._registry.append((compiled_pattern, factory_func, is_pose_capable))
      return factory_func

    return decorator

  @classmethod
  def get_factory(cls, model_name: str) -> tuple[Callable, bool]:
    """Find appropriate factory for the model name"""
    for pattern, factory, is_pose in cls._registry:
      if pattern.match(model_name):
        return factory, is_pose

    raise ValueError(f"No model provider found for '{model_name}'")

  @classmethod
  def create_model(cls, model_name: str, device: str, conf_threshold: float, iou_threshold: float) -> Detector:
    """Create a model instance based on the model name"""
    # Lazy load model types if registry is empty
    if not cls._registry:
      import detection_models.ultralytics  # noqa: F401
      # Add other model types as needed

    factory, _ = cls.get_factory(model_name)
    return factory(model_name, device, conf_threshold, iou_threshold)

  @classmethod
  def is_pose_model(cls, model_name: str) -> bool:
    """Check if a model supports pose detection"""
    for pattern, _, is_pose in cls._registry:
      if pattern.match(model_name) and is_pose:
        return True
    return False

  @classmethod
  def list_registered_patterns(cls) -> list[str]:
    """List all registered model patterns"""
    return [pattern.pattern for pattern, _, _ in cls._registry]

  @classmethod
  def debug_registry(cls, model_name: str) -> None:
    """Print debugging info for model matching"""
    print(f"Trying to match model: '{model_name}'")
    print(f"Available patterns: {cls._registry}")
    for i, (pattern, _factory, _is_pose) in enumerate(cls._registry):
      print(f"Pattern {i}: '{pattern.pattern}' -> Match: {bool(pattern.match(model_name))}")
