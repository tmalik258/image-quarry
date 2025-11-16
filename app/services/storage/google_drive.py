from typing import Optional, Tuple, Dict
from pathlib import Path
from app.utils.logging import setup_logger
from app.config import settings

logger = setup_logger(__name__)

class GoogleDriveService:
    def __init__(self):
        self.enabled = bool(settings.GOOGLE_DRIVE_ENABLED)
        self.credentials_path = settings.GOOGLE_SERVICE_ACCOUNT_JSON
        self.default_folder_id = settings.DRIVE_FOLDER_ID
        self._client = None

    def _ensure_client(self):
        if not self.enabled:
            raise RuntimeError("Google Drive integration disabled")
        if self._client:
            return
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        scopes = ["https://www.googleapis.com/auth/drive.file"]
        creds = Credentials.from_service_account_file(self.credentials_path, scopes=scopes)
        self._client = build("drive", "v3", credentials=creds, cache_discovery=False)

    def status(self) -> Dict[str, Optional[str]]:
        return {
            "enabled": self.enabled,
            "credentials_path": self.credentials_path,
            "default_folder_id": self.default_folder_id,
        }

    def upload_file(self, file_path: str, folder_id: Optional[str] = None) -> Tuple[str, str]:
        self._ensure_client()
        from googleapiclient.http import MediaFileUpload
        service = self._client
        path = Path(file_path)
        metadata = {"name": path.name}
        if folder_id or self.default_folder_id:
            metadata["parents"] = [folder_id or self.default_folder_id]
        media = MediaFileUpload(str(path), resumable=True)
        file = service.files().create(body=metadata, media_body=media, fields="id, webViewLink").execute()
        file_id = file.get("id")
        web_view = file.get("webViewLink")
        logger.info(f"Uploaded to Google Drive file_id={file_id} name={path.name}")
        return file_id, web_view

