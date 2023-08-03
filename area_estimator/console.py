import argparse
import coloredlogs
import copy
import cv2
import functools
import json
import labelbox as lb
import numpy as np
import os
import pprint
import shutil
import tempfile

from collections.abc import Iterable
from dependency_injector.wiring import Provide, inject
from PIL import Image

from . import config
from . import labelbox
from . import labels
from . import line
from . import logger
from . import util
from .containers import AEDeps


def _set_args_overrides(args, container: AEDeps):
    conf = container.config
    config.override_if_non_none(
        conf.labelbox.project_ids, args.project_ids
    )
    config.override_if_non_none(
        conf.area_estimator.log_level, args.log_level
    )


def _add_subparser(*args, **kwargs):
    def _decorator(fn):
        def _add_parsers(parent_subparsers, parents=()):
            subcommand_parser = parent_subparsers.add_parser(
                *args, parents=parents, **kwargs
            )
            subcommand_parser.set_defaults(func=lambda _: subcommand_parser.print_help())
            subparsers = subcommand_parser.add_subparsers(title="action")
            fn(subparsers, parents=parents)
        return _add_parsers
    return _decorator


@_add_subparser("labelbox", help="Actions related to labelbox")
def _add_labelbox_parsers(subparsers: argparse._SubParsersAction, parents=()):
    list_ids_parser = subparsers.add_parser(
        "list-ids", help="List all available sample ids in the project", parents=parents
    )
    list_ids_parser.set_defaults(func=list_sample_ids)

    sample_status_parser = subparsers.add_parser(
        "sample-status-json",
        help="Print out JSON mapping each status to the samples in that status",
        parents=parents
    )
    sample_status_parser.set_defaults(func=sample_status_json)

    refresh_cache_parser = subparsers.add_parser(
        "refresh-cache",
        help="Force an update to the labelbox cache",
        parents=parents,
    )
    refresh_cache_parser.add_argument(
        "--all",
        help="Refresh all projects",
        action="store_true",
        default=False
    )
    refresh_cache_parser.set_defaults(func=refresh_cache)


def build_parser():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--config', '-c', dest='configuration_file',
        default=config.default_config_path()
    )
    parent_parser.add_argument(
        '--project-id', type=str, action='append', dest='project_ids',
        help="The labelbox project id that should be used.",
    )
    parent_parser.add_argument(
        '--refresh-lb-cache', action='store_true', default=False,
        help="Force an update to the labelbox cache"
    )
    parent_parser.add_argument(
        '--model-path', default=None,
        help="Set the path from which model weights should be loaded"
    )
    parent_parser.add_argument(
        '--log-level', '-l', default=None,
        help="Set the log level that will be used throughout area_estimator"
    )
    parent_parser.add_argument(
        '--yolo-log-level', default=None,
        help="Set the log level that will be used for ultralytics yolo"
    )
    parent_parser.add_argument(
        '--fix-cv2', action=util.StoreBooleanAction, default=None, nargs='?'
    )

    parents = [parent_parser]
    parser = argparse.ArgumentParser(parents=parents)
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(title="subcommands")
    _add_labelbox_parsers(subparsers, parents=parents)

    area_parser = subparsers.add_parser("estimate_areas", help="Estimate the areas of all samples")
    area_parser.set_defaults(func=estimate_areas)

    return parser


def main():
    container = AEDeps()

    # Build parser and parse arguments
    parser = build_parser()
    parser.set_defaults(container=container)
    args = parser.parse_args()

    configuration_dict = config.load_config(args.configuration_file)
    container.config.from_dict(
        copy.deepcopy(configuration_dict)
    )
    _set_args_overrides(args, container)
    config.set_defaults(container)

    container.init_resources()
    container.wire(modules=[__name__])

    coloredlogs.install(
        container.config.area_estimator.log_level(), logger=logger,
        fmt=container.config.area_estimator.log_format(),
    )

    # This is done here because we need to install the logger first
    logger.debug(
        f"Loaded values from configuration file {args.configuration_file}:\n"
        f"{pprint.pformat(configuration_dict)}",
    )

    if args.refresh_lb_cache:
        refresh_cache(args)

    logger.debug(f"Parsed args: {pprint.pformat(args)}")

    logger.debug(f"Full config at startup is: {pprint.pformat(container.config())}")

    args.func(args)


@inject
def refresh_cache(
        args,
        projects=Provide[AEDeps.labelbox_projects],
):
    for project in projects:
        project.force_refresh()


@inject
def sample_status_json(
        args, project: labelbox.CachedProject = Provide[AEDeps.labelbox_project]
):
    print(json.dumps(project.get_status_to_ids()))


@inject
def list_sample_ids(args, projects=Provide[AEDeps.labelbox_projects]):
    for project in projects:
        print(f"Project {project._project_id}")
        for sample_id in project.get_ids():
            print(sample_id)


@inject
def estimate_areas(args, project_area_estimator=Provide[AEDeps.project_area_estimator]):
    print(json.dumps(project_area_estimator.get_all()))
