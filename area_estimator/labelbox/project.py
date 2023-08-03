import json
import labelbox
import logging
import os
import uuid

from collections import defaultdict
from typing import Optional


logger = logging.getLogger(__name__)


class CachedProject:

    default_data_to_request = {
        "data_row_details": True,
        "metadata": True,
        "attachments": False,
        "project_details": True,
        "performance_details": False,
        "labels": True,
        "label_details": True,
        "interpolated_frames": False,
        "predictions": True,

    }

    @classmethod
    def from_api_key(cls, api_key: str, *args, **kwargs):
        client = labelbox.Client(api_key=api_key)
        return cls(client, *args, **kwargs)

    def __init__(
            self,
            project_id: str,
            client: labelbox.Client,
            initial_state: str = "To Initial Label",
            cache_directory: Optional[str] = None,
            data_to_request: Optional[dict] = None,
            refresh_on_cache_miss: bool = False,
            **kwargs
    ):
        self._client = client
        self._project_id = project_id
        self._id_to_data = None
        self._data_to_request = data_to_request or self.default_data_to_request
        self._initial_state = initial_state
        self._refresh_on_cache_miss = refresh_on_cache_miss
        if cache_directory:
            os.makedirs(cache_directory, exist_ok=True)
        self._cache_filepath = (
            os.path.join(cache_directory, f"{project_id}.json")
            if cache_directory is not None else None
        )

        if self._cache_filepath is not None:
            try:
                with open(self._cache_filepath, 'r') as f:
                    self._id_to_data = json.loads(f.read())
            except Exception:
                pass

        if self._id_to_data is None:
            self.force_refresh()

    def force_refresh(self):
        self._id_to_data = self._download_labelbox_data(self._cache_filepath)
        if self._cache_filepath is not None:
            with open(self._cache_filepath, 'w') as f:
                f.write(json.dumps(self._id_to_data))

    def _download_labelbox_data(self, cache_filepath):
        project = self._client.get_project(self._project_id)
        export_task = project.export_v2(params=self._data_to_request)

        # Accesing this property waits until we have fulfilled the result so we can check for errors
        export_task.result
        if export_task.errors:
            logger.warn(
                f"Error while processing labelbox task {export_task.errors}"
            )

        return {
            item['data_row']['id']: item
            for item in export_task.result
        }

    def get_ids(self):
        return self._id_to_data.keys()

    def lookup_by_id(self, identifier):
        try:
            return self._id_to_data[identifier]
        except KeyError:
            if self._refresh_on_cache_miss:
                self.force_refresh()
                return self._id_to_data[identifier]
            else:
                raise

    def get_image_url(self, identifier):
        return self.lookup_by_id(identifier)['data_row']['row_data']

    def get_image_dimensions(self, identifier):
        attributes = self.lookup_by_id(identifier)['media_attributes']
        return attributes['width'], attributes['height']

    def get_all_project_annotations(self, identifier):
        labelings = self.lookup_by_id(identifier)['projects'][self._project_id]['labels']
        return [
            data
            for labeling in labelings
            for data in labeling["annotations"]["objects"]
        ]

    def get_available_ids(self):
        return self._id_to_data.keys()

    def get_status(self, identifier):
        workflow_history = self.get_workflow_history(identifier)
        return self._get_status(workflow_history)

    def _get_status(self, workflow_history):
        if not workflow_history:
            return self._initial_state
        last_change = workflow_history[0]
        if last_change['action'] == 'Approve':
            return 'Done'
        if 'next_task_name' not in last_change:
            return self._get_status(workflow_history[1:])
        return last_change['next_task_name']

    def get_status_to_ids(self):
        status_to_ids = defaultdict(lambda: [])
        for sample_id in self.get_ids():
            status = self.get_status(sample_id)
            status_to_ids[status].append(sample_id)
        return status_to_ids

    def upload_labels(self, labels):
        upload_job = labelbox.MALPredictionImport.create_from_objects(
            client=self._client,
            project_id=self._project_id,
            name=f"label_import_job {uuid.uuid4()}",
            predictions=labels
        )
        if upload_job.errors:
            logger.warn(
                f"Error while uploading labelbox model assisted annotations {upload_job.errors}"
            )
        logger.info(
            f"Status of uploads: {upload_job.statuses}"
        )

    def get_workflow_history(self, data_id):
        data = self.lookup_by_id(data_id)
        project_info = data['projects'][self._project_id]
        workflow_history = project_info['project_details']['workflow_history']
        return workflow_history

    def delete_data_row(self, data_row_id):
        self._client(data_row_id)
