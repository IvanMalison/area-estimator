import hashlib
import io
import logging
import os
import requests

from typing import Callable, Optional


logger = logging.getLogger(__name__)


def _bytes_from_filepath(filepath) -> io.BytesIO:
    buf = io.BytesIO()
    with open(filepath, 'rb') as f:
        buf.write(f.read())
        buf.seek(0)
    return buf


class DownloadCache:

    def __init__(self, directory: str, labelbox_session: requests.Session):
        os.makedirs(directory, exist_ok=True)
        self._directory = directory
        self._session = labelbox_session

    def get_existing_filepath(self, url: str) -> Optional[str]:
        filepath = self.filepath_for(url)
        if os.path.isfile(filepath):
            return filepath
        return None

    def filepath_for(self, url: str) -> str:
        filename = hashlib.sha256(url.encode()).hexdigest()
        return os.path.join(self._directory, filename)

    def present(self, url: str) -> bool:
        return self.get_existing_filepath(url) is not None

    def get_buffer(self, url: str) -> Optional[io.BytesIO]:
        filepath = self.get_existing_filepath(url)
        if not self.present(url):
            return None

        return _bytes_from_filepath(filepath)

    def get_or_insert_filepath(
        self, url: str, get_value: Optional[Callable[[str], bytes]] = None
    ) -> str:
        get_value = get_value or self._get_bytes_from_labelbox
        filepath = self.filepath_for(url)
        if os.path.isfile(filepath):
            return filepath

        self._insert(filepath, get_value(url))
        return filepath

    def _get_bytes_from_labelbox(self, url) -> bytes:
        logger.info(f"Downloading from url: {url}")
        response = self._session.get(url)
        response.raise_for_status()
        return response.content

    def _insert(self, path: str, value: bytes):
        with open(path, 'wb') as f:
            f.write(value)
