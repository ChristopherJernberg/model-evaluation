from typing import Callable

from detection.core.interfaces import Detector, ModelConfig


class ModelRegistry:
  # Maps model_name -> (factory_function, is_pose_capable, categories)
  _model_map = {}

  @classmethod
  def register_model(cls, model_name: str, factory_func: Callable, is_pose_capable: bool = False, categories: list[str] = None):
    """Register a specific model with its factory function"""
    categories_list = categories or []
    cls._model_map[model_name] = (factory_func, is_pose_capable, categories_list)

  @classmethod
  def register_models_from_class(cls, model_class, is_pose_capable: bool = False, categories: list[str] = None):
    """Register all models from a model class's SUPPORTED_MODELS dict"""
    # Always include the class name as a category if no categories provided
    class_categories = categories or []
    if not class_categories and hasattr(model_class, "__name__"):
      class_categories = [model_class.__name__]

    for model_name, model_info in model_class.SUPPORTED_MODELS.items():
      model_categories = class_categories.copy()
      if "categories" in model_info and model_info["categories"]:
        model_categories.extend(model_info["categories"])

      cls.register_model(
        model_name,
        lambda name, device, conf, iou, cls=model_class: cls(name, device=device, conf=conf, iou=iou),
        is_pose_capable,
        model_categories,
      )

  @classmethod
  def register_class(cls, is_pose_capable: bool = False, categories: list[str] = None):
    """Decorator to register a model class"""

    def decorator(model_class):
      cls.register_models_from_class(model_class, is_pose_capable, categories)
      return model_class

    return decorator

  @classmethod
  def create_model(cls, model_name: str, device: str, conf_threshold: float, iou_threshold: float) -> Detector:
    """Create a model instance based on the model name"""
    # Lazy load model types if registry is empty
    if not cls._model_map:
      cls._discover_models()

    if model_name not in cls._model_map:
      supported_models = sorted(cls._model_map.keys())
      raise ValueError(f"\nModel '{model_name}' not supported. Choose from:\n{', '.join(supported_models)}")

    factory, _, _ = cls._model_map[model_name]
    return factory(model_name, device, conf_threshold, iou_threshold)

  @classmethod
  def _discover_models(cls):
    """Discover and load all model modules"""
    import importlib
    import os
    import pkgutil

    import detection

    models_path = os.path.join(detection.__path__[0], 'models')
    for _, name, is_pkg in pkgutil.iter_modules([models_path], f"{detection.__name__}.models."):
      try:
        importlib.import_module(name)

        if is_pkg:
          package_path = os.path.join(models_path, name.split('.')[-1])
          for _, subname, _ in pkgutil.iter_modules([package_path]):
            try:
              importlib.import_module(f"{name}.{subname}")
            except ImportError:
              pass
      except ImportError:
        pass

  @classmethod
  def create_from_config(cls, config: ModelConfig) -> Detector:
    """Create a model from a ModelConfig object"""
    return cls.create_model(config.name, config.device, config.conf_threshold, config.iou_threshold)

  @classmethod
  def is_pose_model(cls, model_name: str) -> bool:
    """Check if a model supports pose detection"""
    return cls._model_map.get(model_name, (None, False, []))[1]

  @classmethod
  def list_supported_models(cls) -> list[str]:
    """List all supported model names"""
    return sorted(cls._model_map.keys())

  @classmethod
  def list_models_by_category(cls, category: str = None) -> list[str]:
    """List all models in a category"""
    if category is None:
      return cls.list_supported_models()
    return sorted([name for name, (_, _, categories) in cls._model_map.items() if category in categories])

  @classmethod
  def get_model_categories(cls, model_name: str) -> list[str]:
    """Get all categories a model belongs to"""
    if model_name not in cls._model_map:
      return []
    return cls._model_map[model_name][2]

  @classmethod
  def list_categories(cls) -> list[str]:
    """List all available categories"""
    categories = set()
    for _, _, model_categories in cls._model_map.values():
      categories.update(model_categories)
    return sorted(categories)
