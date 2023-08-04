import logging
import os
import xdg_base_dirs
import yaml


logger = logging.getLogger(__name__)


project_name = os.path.basename(os.path.dirname(__file__)).replace("_", '-')


def set_default_on_config_option(option, default):
    if option() is None:
        option.override(default)


def override_if_has_attr(option, args, attr):
    if hasattr(args, attr):
        return override_if_non_none(option, getattr(args, attr))


def override_if_non_none(option, value):
    if value is not None:
        option.override(value)


def override_with_environment_variable(option, variable):
    value = os.environ.get(variable)
    if value is not None:
        option.override(value)


def default_data_directory():
    return os.path.join(xdg_base_dirs.xdg_data_home(), project_name)


def default_config_directory():
    return os.path.join(xdg_base_dirs.xdg_config_home(), project_name)


def default_config_path(config_directory=None):
    return os.path.join(
        config_directory or default_config_directory(), "config.toml"
    )


def load_config(config_path=None):
    config_path = config_path or default_config_path()
    try:
        import tomllib
        with open(config_path, 'r') as f:
            return tomllib.loads(f.read())
    except Exception as e:
        logger.warn(f"Hit exception trying to load {project_name} config: {e}")
        return {}


def set_defaults(container):
    config = container.config
    data_directory = container.config.area_estimator.data_directory()

    def set_data_directory_default(config_option, *args):
        return set_default_on_config_option(
            config_option, os.path.join(data_directory, *args)
        )

    set_data_directory_default(
        config.labelbox.cache_directory, "labelbox",
    )
    set_data_directory_default(
        config.area_estimator.transformation_cache_directory,
        "image_transformation_cache",
    )
