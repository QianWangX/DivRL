import httpx
import itertools
import os
from fastapi import FastAPI, Request, Response
import uvicorn
import argparse

app = FastAPI()

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    target_base = next(worker_cycle)
    url = f"{target_base}/{path}"
    
    # Forward the request
    body = await request.body()
    resp = await client.request(
        method=request.method,
        url=url,
        content=body,
        headers=dict(request.headers),
        params=dict(request.query_params)
    )
    
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))

if __name__ == "__main__":
    argsparser = argparse.ArgumentParser(description="Load Balancer for MTG Scoring Workers")
    argsparser.add_argument("--port", type=int, default=8098)
    args = argsparser.parse_args()
    host_ip = os.getenv("REWARD_NODE_IP", "0.0.0.0")
    port = args.port
    
    print(f"LOAD BALANCER binding to: {host_ip}:{port}")
    
    # Internal ports where our 3 workers will live
    # WORKER_PORTS = [8101, 8102, 8103]
    WORKER_PORTS = []
    for i in range(1, 4):
        WORKER_PORTS.append(port + i)
    print(f"Worker ports: {WORKER_PORTS}")
    worker_cycle = itertools.cycle([f"http://127.0.0.1:{p}" for p in WORKER_PORTS])

    # Persistent client for high-speed forwarding
    client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=100))

    uvicorn.run(app, host=host_ip, port=port)