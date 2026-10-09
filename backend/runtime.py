import threading
import time
from urllib.parse import urlparse

import requests


_backend_thread: threading.Thread | None = None
_backend_lock = threading.Lock()


def ensure_backend_available(api_url: str) -> str:
    global _backend_thread

    parsed_url = urlparse(api_url)
    local_hosts = {"127.0.0.1", "localhost"}
    can_start_locally = (
        parsed_url.scheme == "http"
        and parsed_url.hostname in local_hosts
        and parsed_url.port is not None
    )

    if not can_start_locally:
        try:
            response = requests.get(
                f"{api_url.rstrip('/')}/health",
                timeout=1,
            )
        except requests.RequestException as error:
            raise RuntimeError(
                f"Could not reach the configured backend at {api_url}."
            ) from error
        if _is_compatible_backend(response):
            return api_url.rstrip("/")
        raise RuntimeError(
            f"The configured backend at {api_url} is not compatible with "
            "this CampusGuide version."
        )

    last_error = "no compatible local backend responded"
    for port in range(parsed_url.port, min(parsed_url.port + 11, 65536)):
        candidate_url = f"http://127.0.0.1:{port}"
        health_url = f"{candidate_url}/health"
        try:
            response = requests.get(health_url, timeout=0.5)
        except requests.RequestException:
            response = None

        if response is not None:
            if _is_compatible_backend(response):
                return candidate_url
            last_error = (
                f"port {port} is occupied by an incompatible university-assistant server"
            )
            continue

        with _backend_lock:
            try:
                response = requests.get(health_url, timeout=0.5)
            except requests.RequestException:
                response = None
            if response is not None:
                if _is_compatible_backend(response):
                    return candidate_url
                last_error = (
                    f"port {port} is occupied by an incompatible university-assistant server"
                )
                continue

            import uvicorn

            from backend.main import app

            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=port,
                    log_level="warning",
                )
            )
            _backend_thread = threading.Thread(
                target=server.run,
                name=f"bu-student-assistant-fastapi-{port}",
                daemon=True,
            )
            _backend_thread.start()

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                response = requests.get(health_url, timeout=1)
            except requests.RequestException as error:
                if _backend_thread is None or not _backend_thread.is_alive():
                    last_error = str(error)
                    break
                time.sleep(0.25)
                continue
            if _is_compatible_backend(response):
                return candidate_url
            last_error = (
                f"the backend started on port {port} but failed its health check"
            )
            break

    raise RuntimeError(
        f"Could not start a compatible FastAPI backend: {last_error}."
    )


def _is_compatible_backend(response: requests.Response) -> bool:
    if not response.ok:
        return False
    try:
        health = response.json()
    except requests.JSONDecodeError:
        return False
    if not isinstance(health, dict):
        return False
    features = health.get("features", [])
    return (
        health.get("status") == "ok"
        and "documents" in features
        and "bundelkhand_university_sources" in features
    )
