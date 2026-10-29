import argparse
import json
import logging
import os
import ssl
from types import SimpleNamespace

import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import fileio
import local_logging

INI_FILE = "app.json"

cfg = json.load(open(INI_FILE, "r"))
directories = SimpleNamespace(**cfg["DIRECTORIES"])
pid_mapping_file = cfg["PID_MAPPING_FILE"]
randomise_id = cfg["RANDOMISED_ID"]

log_file = cfg.get("LOG_FILE")
local_logging.configure_logging(log_file)
fileio.ensure_directories(vars(directories))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def status():
    html_content = """
<html>
    <head>
        <title>DayanTurk status</title>
    </head>
    <body>
        <h1 style="text-align: center">DayanTurk backend is running</h1>
    </body>
</html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/create/{pid}")
async def create_new_ppt(pid: str):
    old_id = await fileio.check_participant(pid_mapping_file, pid)
    if old_id is None:
        new_id = await fileio.create_participant_id(pid_mapping_file, pid, randomise_id)
        await fileio.create_participant_file(directories.INCOMPLETE, new_id)
        logging.info(f"CREATE: generated {new_id} for {pid}")
        return {"id": new_id}
    logging.info(f"CREATE: not generating id for {pid}: already seen")
    return {"id": None}


@app.post("/complete/{ppt_id}")
async def save_data_complete(ppt_id: str, body=Body()):
    await fileio.write_participant_file(directories.INCOMPLETE, ppt_id, body)
    await fileio.move_participant_file(
        directories.INCOMPLETE, directories.COMPLETE, ppt_id
    )
    logging.info(f"COMPLETE: saved completed data for {ppt_id}")


@app.post("/incomplete/{ppt_id}")
async def save_data_incomplete(ppt_id: str, body=Body()):
    await fileio.write_participant_file(directories.INCOMPLETE, ppt_id, body)
    logging.info(f"INCOMPLETE: saved incomplete data for {ppt_id}")


@app.post("/complete_success/{ppt_id}")
async def finalise_success(ppt_id: str):
    await fileio.move_participant_file(
        directories.INCOMPLETE, directories.COMPLETE, ppt_id
    )
    logging.info(f"COMPLETE_SUCCESS: moved data file for {ppt_id}")


@app.post("/complete_failure/{ppt_id}")
async def finalise_failure(ppt_id: str):
    await fileio.move_participant_file(
        directories.INCOMPLETE, directories.INVALID, ppt_id
    )
    logging.info(f"COMPLETE_FAILURE: moved data file for {ppt_id}")


@app.post("/metadata/{ppt_id}")
async def save_metadata(ppt_id: str, body=Body()):
    await fileio.write_participant_file(directories.METADATA, ppt_id, body)
    logging.info(f"METADATA: saved metadata for {ppt_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--local", action="store_true")
    parser.add_argument("-p", "--port", type=int, default=8000)
    args = parser.parse_args()

    if args.local:
        print("Running as a local backend")
        uvicorn.run("main:app", host="127.0.0.1", port=args.port)
    else:
        KEY_FILE = os.environ.get("SSL_KEYFILE")
        if KEY_FILE is None:
            print("Missing 'SSL_KEYFILE' environment variable")
            exit(1)
        CERT_FILE = os.environ.get("SSL_CERTFILE")
        if CERT_FILE is None:
            print("Missing 'SSL_CERTFILE' environment variable")
            exit(1)
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(CERT_FILE, keyfile=KEY_FILE)
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=args.port,
            ssl_certfile=CERT_FILE,
            ssl_keyfile=KEY_FILE,
            ssl_version=ssl.PROTOCOL_TLS_SERVER,
        )
