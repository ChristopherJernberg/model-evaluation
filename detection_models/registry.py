from typing import Callable

from detection_models.detection_interfaces import Detector, ModelConfig


class ModelRegistry:
  # Maps model_name -> (factory_function, is_pose_capable, category)
  _model_map = {}

  @classmethod
  def register_model(cls, model_name: str, factory_func: Callable, is_pose_capable: bool = False, category: str = None):
    """Register a specific model with its factory function"""
    cls._model_map[model_name] = (factory_func, is_pose_capable, category)

  @classmethod
  def register_models_from_class(cls, model_class, is_pose_capable: bool = False, category: str = None):
    """Register all models from a model class's SUPPORTED_MODELS dict"""
    category = category or model_class.__name__
    for model_name in model_class.SUPPORTED_MODELS:
      # Important: Use default args to capture current values
      cls.register_model(model_name, lambda name, device, conf, iou, cls=model_class: cls(name, device=device, conf=conf, iou=iou), is_pose_capable, category)

  @classmethod
  def register_class(cls, is_pose_capable: bool = False, category: str = None):
    """Decorator to register a model class"""

    def decorator(model_class):
      cls.register_models_from_class(model_class, is_pose_capable, category)
      return model_class

    return decorator

  @classmethod
  def create_model(cls, model_name: str, device: str, conf_threshold: float, iou_threshold: float) -> Detector:
    """Create a model instance based on the model name"""
    # Lazy load model types if registry is empty
    if not cls._model_map:
      import detection_models.detr
      import detection_models.ultralytics  # noqa: F401

    if model_name not in cls._model_map:
      supported_models = sorted(cls._model_map.keys())
      raise ValueError(f"\nModel '{model_name}' not supported. Choose from:\n{', '.join(supported_models)}")

    factory, _, _ = cls._model_map[model_name]
    return factory(model_name, device, conf_threshold, iou_threshold)

  @classmethod
  def create_from_config(cls, config: ModelConfig) -> Detector:
    """Create a model from a ModelConfig object"""
    return cls.create_model(config.name, config.device, config.conf_threshold, config.iou_threshold)

  @classmethod
  def is_pose_model(cls, model_name: str) -> bool:
    """Check if a model supports pose detection"""
    return cls._model_map.get(model_name, (None, False))[1]

  @classmethod
  def list_supported_models(cls) -> list[str]:
    """List all supported model names"""
    return sorted(cls._model_map.keys())

  @classmethod
  def list_models_by_category(cls, category: str = None) -> list[str]:
    """List all models in a category"""
    if category is None:
      return cls.list_supported_models()
    return sorted([name for name, (_, _, cat) in cls._model_map.items() if cat == category])
