"""
adapted from https://github.com/xingyaoww/code-act/blob/main/scripts/chat/code_execution/jupyter.py
"""

import asyncio
import os
import re
import uuid
from typing import Optional
from uuid import uuid4

import tornado
from loguru import logger
from tornado.escape import json_decode, json_encode, url_escape
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado.ioloop import PeriodicCallback
from tornado.websocket import websocket_connect


def strip_ansi(o: str) -> str:
    """
    Removes ANSI escape sequences from `o`, as defined by ECMA-048 in
    http://www.ecma-international.org/publications/files/ECMA-ST/Ecma-048.pdf

    # https://github.com/ewen-lbh/python-strip-ansi/blob/master/strip_ansi/__init__.py

    >>> strip_ansi("\\033[33mLorem ipsum\\033[0m")
    'Lorem ipsum'

    >>> strip_ansi("Lorem \\033[38;25mIpsum\\033[0m sit\\namet.")
    'Lorem Ipsum sit\\namet.'

    >>> strip_ansi("")
    ''

    >>> strip_ansi("\\x1b[0m")
    ''

    >>> strip_ansi("Lorem")
    'Lorem'

    >>> strip_ansi('\\x1b[38;5;32mLorem ipsum\\x1b[0m')
    'Lorem ipsum'

    >>> strip_ansi('\\x1b[1m\\x1b[46m\\x1b[31mLorem dolor sit ipsum\\x1b[0m')
    'Lorem dolor sit ipsum'
    """

    # pattern = re.compile(r'/(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]/')
    pattern = re.compile(r"\x1B\[\d+(;\d+){0,2}m")
    stripped = pattern.sub("", o)
    return stripped


class JupyterKernel:
    def __init__(self, convid, port=8789, lang="python"):
        url_suffix = f"localhost:{port}"
        self.base_url = f"http://{url_suffix}"
        self.base_ws_url = f"ws://{url_suffix}"
        self.lang = lang
        self.kernel_id = None
        self.ws = None
        self.convid = convid
        logger.info(
            f"jupyter kernel running at {url_suffix}, conversation id: {self.convid}"
        )

        self.heartbeat_interval = 10000  # 10 seconds
        self.heartbeat_callback = None

    async def _send_heartbeat(self):
        if not self.ws:
            return
        try:
            self.ws.ping()
            # logger.info("Heartbeat sent...")
        except tornado.iostream.StreamClosedError:
            # logger.info("Heartbeat failed, reconnecting...")
            try:
                await self._connect()
            except ConnectionRefusedError:
                logger.error(
                    "ConnectionRefusedError: Failed to reconnect to kernel websocket - Is the kernel still running?"
                )

    async def _connect(self):
        if self.ws:
            self.ws.close()
            self.ws = None

        client = AsyncHTTPClient()
        if not self.kernel_id:
            n_tries = 5
            while n_tries > 0:
                try:
                    response = await client.fetch(
                        "{}/api/kernels".format(self.base_url),
                        method="POST",
                        body=json_encode({"name": self.lang}),
                    )
                    kernel = json_decode(response.body)
                    self.kernel_id = kernel["id"]
                    logger.info(f"kernel created, id: {self.kernel_id}")
                    break
                except Exception as e:
                    # kernels are not ready yet
                    n_tries -= 1
                    logger.error(f"Fail to create kernel: {e}")
                    await asyncio.sleep(1)

            if n_tries == 0:
                raise ConnectionRefusedError("Failed to connect to kernel")

        ws_req = HTTPRequest(
            url="{}/api/kernels/{}/channels".format(
                self.base_ws_url, url_escape(self.kernel_id)
            )
        )
        self.ws = await websocket_connect(ws_req)
        logger.info("Connected to kernel websocket")

        # Setup heartbeat
        if self.heartbeat_callback:
            self.heartbeat_callback.stop()
        self.heartbeat_callback = PeriodicCallback(
            self._send_heartbeat, self.heartbeat_interval
        )
        self.heartbeat_callback.start()

    async def execute(self, code, timeout=60):
        if not self.ws:
            await self._connect()

        msg_id = uuid4().hex
        self.ws.write_message(
            json_encode(
                {
                    "header": {
                        "username": "",
                        "version": "5.0",
                        "session": "",
                        "msg_id": msg_id,
                        "msg_type": "execute_request",
                    },
                    "parent_header": {},
                    "channel": "shell",
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": False,
                        "user_expressions": {},
                        "allow_stdin": False,
                    },
                    "metadata": {},
                    "buffers": {},
                }
            )
        )

        outputs = []

        async def wait_for_messages():
            execution_done = False
            while not execution_done:
                msg = await self.ws.read_message()
                msg = json_decode(msg)
                msg_type = msg["msg_type"]
                parent_msg_id = msg["parent_header"].get("msg_id", None)

                if parent_msg_id != msg_id:
                    continue

                if msg_type == "error":
                    traceback = "\n".join(msg["content"]["traceback"])
                    outputs.append(traceback)
                    execution_done = True
                elif msg_type == "stream":
                    outputs.append(msg["content"]["text"])
                    logger.info(
                        "intermediate jupyter run output\n"
                        + msg["content"]["text"].strip()
                    )
                elif msg_type in ["execute_result", "display_data"]:
                    outputs.append(msg["content"]["data"]["text/plain"])
                    if "image/png" in msg["content"]["data"]:
                        # use markdown to display image (in case of large image)
                        # outputs.append(f"\n<img src=\"data:image/png;base64,{msg['content']['data']['image/png']}\"/>\n")
                        outputs.append(
                            f"![image](data:image/png;base64,{msg['content']['data']['image/png']})"
                        )

                elif msg_type == "execute_reply":
                    execution_done = True
            return execution_done

        async def interrupt_kernel():
            client = AsyncHTTPClient()
            interrupt_response = await client.fetch(
                f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt",
                method="POST",
                body=json_encode({"kernel_id": self.kernel_id}),
            )
            logger.info(f"Kernel interrupted: {interrupt_response}")

        try:
            execution_done = await asyncio.wait_for(wait_for_messages(), timeout)
        except asyncio.TimeoutError:
            await interrupt_kernel()
            return f"[Execution timed out ({timeout} seconds).]"

        if not outputs and execution_done:
            ret = "[Code executed successfully with no output]"
        else:
            ret = "".join(outputs)

        # Remove ANSI
        ret = strip_ansi(ret)

        if os.environ.get("DEBUG", False):
            logger.info(f"OUTPUT:\n{ret}")
        return ret

    async def shutdown_async(self):
        if self.kernel_id:
            kid = self.kernel_id
            client = AsyncHTTPClient()
            await client.fetch(
                "{}/api/kernels/{}".format(self.base_url, self.kernel_id),
                method="DELETE",
            )
            self.kernel_id = None
            if self.ws:
                self.ws.close()
                self.ws = None
            logger.info(f"kernel with id: {kid}  is closed")


async def _test():
    convid = str(uuid.uuid4())
    jk = JupyterKernel(convid)
    code = """
import psutil
import platform

# Get CPU information
cpu_count = psutil.cpu_count(logical=False)
cpu_count_logical = psutil.cpu_count(logical=True)
cpu_freq = psutil.cpu_freq()

# Get memory information
memory = psutil.virtual_memory()

# Get disk information
disk = psutil.disk_usage('.')

# Get system information
system = platform.system()
release = platform.release()
version = platform.version()
machine = platform.machine()
processor = platform.processor()

import os
import multiprocessing

def get_usable_cpu_count():
    # SLURM
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    elif "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        return int(os.environ["SLURM_JOB_CPUS_PER_NODE"])

    # SGE / Grid Engine
    elif "NSLOTS" in os.environ:
        return int(os.environ["NSLOTS"])
    
    # LSF
    elif "LSB_DJOB_NUMPROC" in os.environ:
        return int(os.environ["LSB_DJOB_NUMPROC"])
    
    # PBS / Torque
    elif "PBS_NP" in os.environ:
        return int(os.environ["PBS_NP"])

    # Default to available logical CPUs
    try:
        import psutil
        return len(psutil.Process().cpu_affinity())
    except (ImportError, AttributeError):
        return multiprocessing.cpu_count()

print("usable CPU num:",get_usable_cpu_count())
print(f"System: {system} {release} {version}")
print(f"Machine: {machine}")
print(f"Processor: {processor}")
print(f"CPU Cores (Physical): {cpu_count}")
print(f"CPU Cores (Logical): {cpu_count_logical}")
print(f"CPU Frequency: {cpu_freq.max} MHz")
print(f"Total Memory: {memory.total / (1024**3):.2f} GB")
print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
print(f"Memory Usage: {memory.percent}%")
print(f"Total Disk Space: {disk.total / (1024**3):.2f} GB")
print(f"Used Disk Space: {disk.used / (1024**3):.2f} GB")
print(f"Free Disk Space: {disk.free / (1024**3):.2f} GB")
    """
    print(await jk.execute(code))

    print(await jk.execute("!lscpu"))

    await jk.shutdown_async()

    return


if __name__ == "__main__":
    asyncio.run(_test())  # uv run -m utils.jupyter
