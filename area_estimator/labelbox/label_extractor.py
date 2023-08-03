from typing import Optional, Sequence

from .download_cache import DownloadCache
from .. import labels


class LabelExtractor:

    def __init__(
            self,
            download_cache: DownloadCache,
            limit_to_types: Optional[Sequence[str]] = None,
            **kwargs
    ):
        self._download_cache = download_cache
        self._name_to_constructor = {
            name: (
                self._build_mask_with_constructor(klass) if
                issubclass(klass, labels.FileSegmentationMask)
                else klass.from_labelbox_data
            )
            for name, klass in labels.Annotation.name_to_class.items()
        }
        self._relevant_types = limit_to_types or list(self._name_to_constructor.keys())

    def get_objects_by_type(self, objects):
        objects_by_type = {}
        for obj in objects:
            obj_type = obj["name"]
            if obj_type in self._relevant_types:
                objects_by_type.setdefault(obj["name"], []).append(
                    self._get_value_for_object(obj)
                )
        return objects_by_type

    def _build_mask_with_constructor(self, constructor):
        def build(labelbox_annotation):
            return constructor(self._get_mask_filepath(labelbox_annotation["mask"]["url"]))
        return build

    def _get_mask_filepath(self, url: str) -> str:
        return self._download_cache.get_or_insert_filepath(
            url,
        )

    def _get_constructor_for_object(self, obj):
        return self._name_to_constructor.get(obj["name"], lambda obj: obj)

    def _get_value_for_object(self, obj):
        return self._get_constructor_for_object(obj)(obj)
