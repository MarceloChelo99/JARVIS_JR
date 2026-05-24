"""Google Calendar client (read + create + update + delete)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/calendar"]

REPO_ROOT = Path(__file__).resolve().parents[3]
CREDENTIALS_DIR = REPO_ROOT / "credentials"
TOKEN_PATH = CREDENTIALS_DIR / "gcal_token.json"
CLIENT_SECRET_PATH = CREDENTIALS_DIR / "gcal_client_secret.json"


class CalendarError(RuntimeError):
    """Raised when the calendar client cannot operate."""


def _load_creds() -> Credentials:
    if not TOKEN_PATH.exists():
        raise CalendarError(
            f"No Google Calendar token at {TOKEN_PATH}. "
            "Run `uv run python scripts/setup_gcal.py` first."
        )
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
        else:
            raise CalendarError(
                "Stored credentials are invalid and cannot be refreshed. "
                "Re-run scripts/setup_gcal.py."
            )
    return creds


class GoogleCalendar:
    """Thin wrapper around the Google Calendar v3 API."""

    def __init__(
        self,
        calendar_id: str = "primary",
        default_duration_minutes: int = 30,
        default_timezone: str | None = None,
    ):
        self.calendar_id = calendar_id
        self.default_duration_minutes = default_duration_minutes
        self.default_timezone = default_timezone
        self._service = build("calendar", "v3", credentials=_load_creds(), cache_discovery=False)

    def create_event(
        self,
        title: str,
        start: str,
        end: str | None = None,
        location: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create an event. `start`/`end` are ISO 8601 strings."""
        if end is None:
            end = (
                datetime.fromisoformat(start)
                + timedelta(minutes=self.default_duration_minutes)
            ).isoformat()
        body: dict[str, Any] = {
            "summary": title,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
        }
        if self.default_timezone:
            body["start"]["timeZone"] = self.default_timezone
            body["end"]["timeZone"] = self.default_timezone
        if location:
            body["location"] = location
        if description:
            body["description"] = description
        return self._service.events().insert(calendarId=self.calendar_id, body=body).execute()

    def list_events(self, start: str, end: str, max_results: int = 10) -> list[dict[str, Any]]:
        """List events in [start, end). ISO 8601 strings."""
        # Google requires Zulu time for timeMin/timeMax; if a tz-aware string
        # is given, isoformat preserves the offset which the API also accepts.
        resp = (
            self._service.events()
            .list(
                calendarId=self.calendar_id,
                timeMin=start,
                timeMax=end,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        return resp.get("items", [])

    def update_event(self, event_id: str, **changes: Any) -> dict[str, Any]:
        """Patch an event. `changes` keys mirror the Google Calendar v3 schema."""
        return (
            self._service.events()
            .patch(calendarId=self.calendar_id, eventId=event_id, body=changes)
            .execute()
        )

    def delete_event(self, event_id: str) -> None:
        self._service.events().delete(
            calendarId=self.calendar_id, eventId=event_id
        ).execute()
