
import sys
import logging
import typing
import importlib


def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s'
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

# https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
def load_obj(obj_path: str, default_obj_path: str = "", name: str = None) -> typing.Any:
    
    """
    Used to Load Objects from config files.
    Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """

    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    # obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)

    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )

    return getattr(module_obj, obj_name)
    
