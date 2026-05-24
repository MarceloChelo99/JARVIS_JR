"""One-time Google Calendar OAuth setup.

Before running this script:

  1. Go to https://console.cloud.google.com/
  2. Create a project (or reuse one).
  3. Enable the Google Calendar API for the project.
  4. APIs & Services → Credentials → Create Credentials → OAuth client ID.
       Application type: "Desktop app".
  5. Download the resulting JSON.
  6. Save it as:  credentials/gcal_client_secret.json
  7. Run this script. A browser tab will open for you to grant access. The
     token is then saved to credentials/gcal_token.json and refreshed
     automatically thereafter.
"""

from __future__ import annotations

import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

from jarvis_jr.tools.calendar import CLIENT_SECRET_PATH, SCOPES, TOKEN_PATH


def main() -> int:
    if not CLIENT_SECRET_PATH.exists():
        print(f"Missing client secret file: {CLIENT_SECRET_PATH}")
        print("See the docstring at the top of this script for setup steps.")
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(port=0)

    Path(TOKEN_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(TOKEN_PATH).write_text(creds.to_json())
    print(f"Token saved to {TOKEN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
