from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel

from spatial_ingestion.batch_normalization.normalizer import BatchNormalizer
from spatial_ingestion.config import MAX_UPLOAD_FILE_BYTES
from spatial_ingestion.ingestion_gateway.auth import AuthContext, AuthProvider
from spatial_ingestion.ingestion_gateway.rate_limit import RateLimiter
from spatial_ingestion.live_stream.manager import (
    LiveStreamManager,
    StreamLimitExceeded,
    StreamOwnershipError,
)
from spatial_ingestion.media_classifier.router import MediaClassifierRouter, MediaItemDescriptor
from spatial_ingestion.metadata.schema import UnifiedSpatialIngestionSchema
from spatial_ingestion.storage.object_store import ObjectStore


class StreamConnectRequest(BaseModel):
    transport: str
    stream_id: str | None = None


class GatewayState:
    def __init__(self) -> None:
        self.auth = AuthProvider()
        self.rate_limiter = RateLimiter()
        self.router = MediaClassifierRouter()
        self.batch_normalizer = BatchNormalizer()
        self.live_streams = LiveStreamManager()
        self.object_store = ObjectStore()


class UploadTooLargeError(Exception):
    """Raised when an upload exceeds the configured ingestion size limit."""


async def auth_context(
    request: Request,
    authorization: str | None = Header(default=None),
) -> AuthContext:
    state = _gateway_state(request)
    client_host = request.client.host if request.client else None
    context = await state.auth.authenticate(authorization, client_host)
    rate = await state.rate_limiter.check(context.subject)
    if not rate.allowed:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    return context


def create_app() -> FastAPI:
    app = FastAPI(title="Spatial Ingestion Phase 1", version="0.1.0")
    app.state.gateway_state = GatewayState()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/ingest/uploads", response_model=UnifiedSpatialIngestionSchema)
    async def ingest_uploads(
        request: Request,
        files: list[UploadFile] = File(...),
        _: AuthContext = Depends(auth_context),
    ) -> UnifiedSpatialIngestionSchema:
        state = _gateway_state(request)
        descriptors = [
            MediaItemDescriptor(
                filename=file.filename or "upload",
                mime_type=file.content_type,
            )
            for file in files
        ]
        decision = state.router.classify_static(descriptors)
        if decision.input_type.value == "unknown":
            raise HTTPException(status_code=415, detail={"routing_decision": decision.__dict__})

        with TemporaryDirectory(prefix="spatial_ingest_") as temp_dir:
            paths: list[Path] = []
            for index, file in enumerate(files):
                filename = Path(file.filename or f"upload_{index}").name
                target = Path(temp_dir) / filename
                try:
                    await _stream_upload_to_disk(file, target)
                except UploadTooLargeError as exc:
                    raise HTTPException(status_code=413, detail=str(exc)) from exc
                paths.append(target)

            original_uris = {
                path: state.object_store.put_file(path, "originals")
                for path in paths
            }
            try:
                return state.batch_normalizer.normalize(
                    paths,
                    decision,
                    original_uris=original_uris,
                )
            except Exception:
                for uri in original_uris.values():
                    state.object_store.delete_uri(uri)
                raise

    @app.post("/v1/ingest/streams/connect", response_model=UnifiedSpatialIngestionSchema)
    async def connect_stream(
        request: Request,
        stream_request: StreamConnectRequest,
        auth: AuthContext = Depends(auth_context),
    ) -> UnifiedSpatialIngestionSchema:
        state = _gateway_state(request)
        decision = state.router.classify_stream(stream_request.transport, stream_request.stream_id)
        if decision.input_type.value == "unknown":
            raise HTTPException(status_code=415, detail={"routing_decision": decision.__dict__})

        try:
            payload = state.live_streams.open_stream(
                stream_request.stream_id,
                owner_subject=auth.subject,
            )
        except StreamOwnershipError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except StreamLimitExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

        payload.compute_priority_score = decision.priority_score
        payload.metadata.update(
            {
                "transport": stream_request.transport.lower(),
            }
        )
        return payload

    @app.websocket("/v1/ingest/streams/{stream_id}/frames")
    async def stream_frames(websocket: WebSocket, stream_id: str) -> None:
        state = _gateway_state(websocket)
        authorization = websocket.headers.get("authorization") or websocket.query_params.get("token")
        client_host = websocket.client.host if websocket.client else None
        auth = await state.auth.authenticate(authorization, client_host)
        rate = await state.rate_limiter.check(auth.subject)
        if not rate.allowed:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        if not state.live_streams.has_stream(stream_id):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        if not state.live_streams.is_owner(stream_id, auth.subject):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await websocket.accept()

        try:
            while True:
                payload = await websocket.receive_bytes()
                decision = state.live_streams.push_encoded_frame(stream_id, payload)
                await websocket.send_json(
                    {
                        "accepted": decision.accepted,
                        "action": decision.action,
                        "dropped_frames": decision.dropped_frames,
                    }
                )
        except WebSocketDisconnect:
            return
        finally:
            state.live_streams.close_stream(stream_id, auth.subject)

    return app


async def _stream_upload_to_disk(file: UploadFile, target: Path) -> None:
    total = 0
    with target.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_FILE_BYTES:
                raise UploadTooLargeError("uploaded file too large")
            output.write(chunk)


def _gateway_state(request: Request | WebSocket) -> GatewayState:
    return request.app.state.gateway_state
