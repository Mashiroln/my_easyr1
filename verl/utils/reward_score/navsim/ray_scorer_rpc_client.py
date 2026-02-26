"""Tiny client helpers for the Ray scorer UDS RPC protocol (no HTTP)."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import struct
from typing import Any, Dict, List, Optional

import orjson


def _pack_frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        data = sock.recv(remaining)
        if not data:
            raise ConnectionError("socket closed")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _recv_frame(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, 4)
    (n,) = struct.unpack(">I", header)
    if n == 0:
        return b""
    return _recv_exact(sock, n)


def make_score_payload(
    *,
    token: str,
    trajectory_se2: List[List[float]],
    req_id: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    return {
        "token": token,
        "trajectory_se2": trajectory_se2,
        "verbose": bool(verbose),
        "req_id": int(req_id),
    }


def make_score_group_payload(
    *,
    token: str,
    trajectories_se2: List[List[List[float]]],
    req_id: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    return {
        "token": token,
        "trajectories_se2": trajectories_se2,
        "verbose": bool(verbose),
        "req_id": int(req_id),
    }


def score_single_frame(
    *,
    uds_path: str,
    token: str,
    trajectory_se2: List[List[float]],
    verbose: bool = False,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    with RpcClientSync(uds_path=uds_path, timeout_s=timeout_s) as c:
        return c.score_single_frame(token=token, trajectory_se2=trajectory_se2, verbose=verbose)


class RpcClientSync:
    def __init__(self, uds_path: str = "/tmp/simscale_scorer_rpc.sock", timeout_s: float = 120.0):
        self._uds_path = uds_path
        self._timeout_s = float(timeout_s)
        self._sock: Optional[socket.socket] = None
        self._next_id = 1

    def connect(self) -> "RpcClientSync":
        if self._sock is not None:
            return self
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout_s)
        sock.connect(self._uds_path)
        self._sock = sock
        return self

    def close(self) -> None:
        if self._sock is not None:
            with contextlib.suppress(Exception):
                self._sock.close()
        self._sock = None

    def __enter__(self) -> "RpcClientSync":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._sock is None:
            self.connect()
        assert self._sock is not None

        if "req_id" not in payload:
            payload = dict(payload)
            payload["req_id"] = int(self._next_id)
            self._next_id += 1

        raw = orjson.dumps(payload)
        self._sock.sendall(_pack_frame(raw))

        resp_raw = _recv_frame(self._sock)
        msg = orjson.loads(resp_raw)
        if not isinstance(msg, dict):
            raise RuntimeError("invalid response")

        if msg.get("ok") is True:
            result = msg.get("result")
            if isinstance(result, dict):
                return result
            return {"_result": result}

        raise RuntimeError(str(msg.get("error", "unknown error")))

    def score(
        self,
        *,
        token: str,
        trajectory_se2: List[List[float]],
        verbose: bool = False,
        req_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if req_id is None:
            req_id = int(self._next_id)
            self._next_id += 1
        payload = make_score_payload(token=token, trajectory_se2=trajectory_se2, req_id=req_id, verbose=verbose)
        return self.request(payload)

    def score_group(
        self,
        *,
        token: str,
        trajectories_se2: List[List[List[float]]],
        verbose: bool = False,
        req_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if req_id is None:
            req_id = int(self._next_id)
            self._next_id += 1
        payload = make_score_group_payload(
            token=token,
            trajectories_se2=trajectories_se2,
            req_id=req_id,
            verbose=verbose,
        )
        return self.request(payload)

    def score_single_frame(self, *, token: str, trajectory_se2: List[List[float]], verbose: bool = False) -> Dict[str, Any]:
        return self.score(token=token, trajectory_se2=trajectory_se2, verbose=verbose, req_id=None)


async def _read_frame(reader: asyncio.StreamReader) -> Optional[bytes]:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    (n,) = struct.unpack(">I", header)
    if n == 0:
        return b""
    return await reader.readexactly(n)


async def _write_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    writer.write(struct.pack(">I", len(payload)))
    if payload:
        writer.write(payload)
    await writer.drain()


class RpcClientAsync:
    def __init__(self, uds_path: str = "/tmp/simscale_scorer_rpc.sock", timeout_s: float = 120.0):
        self._uds_path = uds_path
        self._timeout_s = float(timeout_s)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._write_lock = asyncio.Lock()
        self._pending: Dict[int, asyncio.Future[Dict[str, Any]]] = {}
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._next_id = 1

    async def connect(self) -> "RpcClientAsync":
        if self._reader is not None:
            return self
        self._reader, self._writer = await asyncio.open_unix_connection(self._uds_path)
        self._reader_task = asyncio.create_task(self._read_loop())
        return self

    async def close(self) -> None:
        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
                await self._writer.wait_closed()
        self._reader = None
        self._writer = None

        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(Exception):
                await self._reader_task
        self._reader_task = None

    async def __aenter__(self) -> "RpcClientAsync":
        return await self.connect()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _read_loop(self) -> None:
        assert self._reader is not None
        while True:
            frame = await _read_frame(self._reader)
            if frame is None:
                break
            try:
                msg = orjson.loads(frame)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            req_id = msg.get("req_id")
            if not isinstance(req_id, int):
                continue
            fut = self._pending.pop(req_id, None)
            if fut is not None and not fut.done():
                fut.set_result(msg)

        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("RPC connection closed"))
        self._pending.clear()

    async def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._writer is None:
            await self.connect()
        assert self._writer is not None

        if "req_id" not in payload:
            payload = dict(payload)
            payload["req_id"] = int(self._next_id)
            self._next_id += 1

        req_id = payload.get("req_id")
        if not isinstance(req_id, int):
            raise ValueError("payload.req_id must be int")

        fut: asyncio.Future[Dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut

        raw = orjson.dumps(payload)
        async with self._write_lock:
            await _write_frame(self._writer, raw)

        msg = await asyncio.wait_for(fut, timeout=self._timeout_s)
        if msg.get("ok") is True:
            result = msg.get("result")
            if isinstance(result, dict):
                return result
            return {"_result": result}

        raise RuntimeError(str(msg.get("error", "unknown error")))

    async def score(
        self,
        *,
        token: str,
        trajectory_se2: List[List[float]],
        verbose: bool = False,
        req_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if req_id is None:
            req_id = int(self._next_id)
            self._next_id += 1
        payload = make_score_payload(token=token, trajectory_se2=trajectory_se2, req_id=req_id, verbose=verbose)
        return await self.request(payload)

    async def score_group(
        self,
        *,
        token: str,
        trajectories_se2: List[List[List[float]]],
        verbose: bool = False,
        req_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if req_id is None:
            req_id = int(self._next_id)
            self._next_id += 1
        payload = make_score_group_payload(
            token=token,
            trajectories_se2=trajectories_se2,
            req_id=req_id,
            verbose=verbose,
        )
        return await self.request(payload)

    async def score_single_frame(self, *, token: str, trajectory_se2: List[List[float]], verbose: bool = False) -> Dict[str, Any]:
        return await self.score(token=token, trajectory_se2=trajectory_se2, verbose=verbose, req_id=None)
