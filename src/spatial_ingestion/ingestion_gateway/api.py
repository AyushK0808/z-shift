from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

from spatial_ingestion.batch_normalization.normalizer import BatchNormalizer
from spatial_ingestion.ingestion_gateway.auth import AuthContext, AuthProvider
from spatial_ingestion.ingestion_gateway.rate_limit import RateLimiter
from spatial_ingestion.live_stream.manager import LiveStreamManager
from spatial_ingestion.media_classifier.router import MediaClassifierRouter, MediaItemDescriptor
from spatial_ingestion.metadata.schema import UnifiedSpatialIngestionSchema


class StreamConnectRequest(BaseModel):
    transport: str
    stream_id: str | None = None
    rtsp_url: str | None = None
    webrtc_offer_sdp: str | None = None


class GatewayState:
    def __init__(self) -> None:
        self.auth = AuthProvider()
        self.rate_limiter = RateLimiter()
        self.router = MediaClassifierRouter()
        self.batch_normalizer = BatchNormalizer()
        self.live_streams = LiveStreamManager()


state = GatewayState()


async def auth_context(authorization: str | None = Header(default=None)) -> AuthContext:
    context = await state.auth.authenticate(authorization)
    rate = await state.rate_limiter.check(context.subject)
    if not rate.allowed:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    return context


def create_app() -> FastAPI:
    app = FastAPI(title="Spatial Ingestion Phase 1", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/ingest/uploads", response_model=UnifiedSpatialIngestionSchema)
    async def ingest_uploads(
        files: list[UploadFile] = File(...),
        _: AuthContext = Depends(auth_context),
    ) -> UnifiedSpatialIngestionSchema:
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
                target.write_bytes(await file.read())
                paths.append(target)

            return state.batch_normalizer.normalize(paths, decision)

    @app.post("/v1/ingest/streams/connect", response_model=UnifiedSpatialIngestionSchema)
    async def connect_stream(
        request: StreamConnectRequest,
        _: AuthContext = Depends(auth_context),
    ) -> UnifiedSpatialIngestionSchema:
        decision = state.router.classify_stream(request.transport, request.stream_id)
        if decision.input_type.value == "unknown":
            raise HTTPException(status_code=415, detail={"routing_decision": decision.__dict__})

        payload = state.live_streams.open_stream(request.stream_id)
        payload.compute_priority_score = decision.priority_score
        payload.metadata.update(
            {
                "transport": request.transport.lower(),
                "rtsp_url": request.rtsp_url,
                "webrtc_offer_sdp_received": bool(request.webrtc_offer_sdp),
            }
        )
        return payload

    @app.websocket("/v1/ingest/streams/{stream_id}/frames")
    async def stream_frames(websocket: WebSocket, stream_id: str) -> None:
        await websocket.accept()
        if stream_id not in state.live_streams._streams:
            state.live_streams.open_stream(stream_id)

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

    return app

