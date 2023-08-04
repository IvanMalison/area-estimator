import labelbox
import logging
import operator
import os

from dependency_injector import providers, containers

from . import area
from . import config as defaults
from . import labelbox as rb_labelbox
from . import util
from . import spreadsheet


class AEDeps(containers.DeclarativeContainer):

    config = providers.Configuration(default={
        "area_estimator": {
            "data_directory": defaults.default_data_directory(),
            "log_level": logging.INFO,
            "log_format": '%(asctime)s %(name)s %(levelname)s %(message)s',
            "fix_cv2": True,
            "field_of_view": 84,
            "default_height": 300,
        },
        "google": {
            "api_key_path": os.path.join(
                defaults.default_config_directory(), "google", "credentials.json",
            ),
        },
        "labelbox": {
            "data_to_request": {
                "data_row_details": True,
                "metadata": True,
                "attachments": False,
                "project_details": True,
                "performance_details": False,
                "label_details": True,
                "interpolated_frames": False,
                "labels": True,
                "predictions": True,
            }
        },
    })

    labelbox_client = providers.Singleton(
        labelbox.Client,  # type: ignore
        api_key=config.labelbox.api_key
    )

    labelbox_headers = providers.Singleton(
        lambda client: client.headers,
        labelbox_client,
    )

    labelbox_session = providers.Singleton(
        util.new_session_with_headers,
        labelbox_headers
    )

    labelbox_project_factory = providers.Factory(
        rb_labelbox.CachedProject,
        cache_directory=config.labelbox.cache_directory,
        client=labelbox_client,
    )

    labelbox_projects = providers.Singleton(
        lambda f, project_ids, **kwargs: [f(project_id=pid, **kwargs) for pid in project_ids],
        labelbox_project_factory.provider,
        config.labelbox.project_ids,
        data_to_request=config.labelbox.data_to_request,
    )

    labelbox_project = providers.Singleton(operator.itemgetter(0), labelbox_projects)

    download_cache = providers.Singleton(
        rb_labelbox.DownloadCache,
        providers.Singleton(
            lambda lbdir: os.path.join(lbdir, "download_cache"),
            config.labelbox.cache_directory
        ),
        labelbox_session,
    )

    label_extractor_factory = providers.Factory(
        rb_labelbox.LabelExtractor,
        download_cache=download_cache,
    )

    label_extractor = providers.Singleton(label_extractor_factory)

    calculator = providers.Singleton(
        area.AreaCalculator,
        config.area_estimator.field_of_view
    )

    project_area_estimator = providers.Singleton(
        area.ProjectAreaEstimator,
        calculator,
        labelbox_projects,
        label_extractor_factory,
    )

    spreadsheet_updater = providers.Singleton(
        spreadsheet.GoogleSheetUpdater,
        project_area_estimator,
        config.google.api_key_path,
        config.google.spreadsheet_url,
    )
