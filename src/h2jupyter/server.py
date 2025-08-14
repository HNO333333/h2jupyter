import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Optional

import yaml
from fabric2 import Connection
from fabric2.tunnels import TunnelManager
from loguru import logger as base
from paramiko import Channel
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# --- global vars
current_dir = os.path.curdir
thread_stop_event = threading.Event()
remote_host = ""
avail_port = 8789


# --- signal capturing
def exit_handler(signum, frame):
    logger.info("Exiting...")
    exit()


signal.signal(signal.SIGINT, exit_handler)
# --- logger setting
base.remove()

logger = deepcopy(base)
qsub_stdout_logger = deepcopy(base)
tunnel_stdout_logger = deepcopy(base)

logger.add(sys.stdout, level="INFO")
qsub_stdout_logger.add(
    sys.stdout,
    level="INFO",
    format="{message}",
)
tunnel_stdout_logger.add(
    sys.stdout,
    level="INFO",
    format="[tunnel] {message}",
)

logfile_stream = open(os.path.join(current_dir, "server.log"), "a")


# --- helpers
def get_current_time_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_ansi_escape(s: str) -> str:
    # ANSI escape sequences usually start with \x1b (ESC) followed by '[' and some characters ending with a letter
    ansi_escape = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
    return ansi_escape.sub("", s)


def get_session_queue(conn: Connection):
    session = conn.transport.open_session()
    session.get_pty()
    session.invoke_shell()
    q = queue.Queue()
    return session, q


def readwhile(q: queue.Queue, condition_func, timeout=10, verbose=False):
    """
    Read from a stream until a condition is met or a timeout occurs.
    """
    start = time.time()
    flag = False
    while True:
        try:
            data: str = q.get(timeout=0.1)
            for line in data.split("\n"):
                if verbose:
                    logger.info(repr(line.strip()))
                if condition_func(line.strip()):
                    flag = True
                    break
            if flag:
                break
        except queue.Empty:
            if timeout:
                if time.time() - start > timeout:
                    raise TimeoutError("No output received for too long.")
            else:
                time.sleep(0.1)
        except Exception as e:
            logger.error(e)
            break
    return


class ParallelEnv(Enum):
    SHARED = "shared"
    DISTRIBUTED = "dc\\*"


class PythonVersion(Enum):
    PYTHON_3_10_18 = "python/3.10.18"
    PYTHON_3_11_12 = "python/3.11.12"
    PYTHON_3_12_9 = "python/3.12.9"


class CPUArch(Enum):
    AMD_EPYC_7642 = "amd-epyc-7642"
    INTEL_E5_2650 = "intel-E5-2650"
    INTEL_E5_2650V2 = "intel-E5-2650v2"
    INTEL_E5_2650V4 = "intel-E5-2650v4"
    INTEL_E5_2670 = "intel-E5-2670"
    INTEL_E5_2670V2 = "intel-E5-2670v2"
    INTEL_E5_2670V3 = "intel-E5-2670v3"
    INTEL_E5_2697AV4 = "intel-E5-2697Av4"
    INTEL_E5_4620V2 = "intel-E5-4620v2"
    INTEL_E7_8890V4 = "intel-E7-8890v4"
    INTEL_GOLD_5118 = "intel-gold-5118"
    INTEL_GOLD_5218 = "intel-gold-5218"
    INTEL_GOLD_6140 = "intel-gold-6140"
    INTEL_GOLD_6240 = "intel-gold-6240"
    LX_AMD64 = "lx-amd64"
    INTEL_WILDCARD = "intel\\*"
    INTEL_GOLD_WILDCARD = "intel-gold\\*"


class GPUType(Enum):
    K40 = "K40"
    P4 = "P4"
    GTX1080TI = "GTX1080Ti"
    RTX2080TI = "RTX2080Ti"
    V100 = "V100"
    A100 = "A100"
    A6000 = "A6000"
    H100 = "H100"
    L40S = "L40S"
    NONE = ""


class CudaVersion(Enum):
    CUDA_9_2 = "cuda/9.2"
    CUDA_10_0 = "cuda/10.0"
    CUDA_10_2 = "cuda/10.2"
    CUDA_11_0 = "cuda/11.0"
    CUDA_11_3 = "cuda/11.3"
    CUDA_11_7 = "cuda/11.7"
    CUDA_11_8 = "cuda/11.8"
    CUDA_12_3 = "cuda/12.3"


class HPCServerConfig(BaseModel):
    # --- ssh
    hostname: Optional[str] = Field(
        default=None, description="HPC server ssh hostname (check ~/.ssh/config)"
    )
    username: str = Field(default="", description="Username for the HPC server")
    sshlogs: bool = Field(default=False, description="Whether to enable SSH logging")
    usetunnel: bool = Field(default=False, description="Whether to open SSH tunnel")
    # --- l options
    timeinhours: Optional[int] = Field(
        default=2, description="Time in hours for the job allocation"
    )
    memoryingb: Optional[int] = Field(default=5, description="Memory allocation in GB")
    arch: Optional[CPUArch] = Field(
        default=CPUArch.INTEL_GOLD_WILDCARD, description="CPU architecture requirement"
    )
    highp: bool = Field(default=False, description="whether run on owned node")
    exclusive: bool = Field(default=False, description="whether run in exclusive mode")
    # --- pe options
    parenv: ParallelEnv = Field(
        default=ParallelEnv.SHARED, description="Parallel environment type"
    )
    numberofslots: int = Field(
        default=1, description="Number of computing cores to use"
    )
    # --- module
    pythonver: PythonVersion = Field(
        default=PythonVersion.PYTHON_3_11_12, description="Python version to use"
    )
    gccver: Optional[str] = Field(
        default="gcc/11.3.0", description="GCC compiler version"
    )
    mods: Optional[str] = Field(
        default="", description="Modules to load, separated by ','"
    )
    # --- GPU
    usegpu: bool = Field(default=False, description="Whether to use GPU resources")
    gputype: Optional[GPUType] = Field(
        default=GPUType.NONE, description="Type of GPU to use"
    )
    cudaver: Optional[CudaVersion] = Field(
        default=CudaVersion.CUDA_11_8, description="CUDA version"
    )
    cudanum: int = Field(default=1, description="Number of CUDA devices to use")
    # --- env
    cve: Optional[str] = Field(default="", description="conda environment")
    pve: Optional[str] = Field(default="", description="Python virtual environment")
    port: int = Field(default=8789, description="Port number for the server")
    directory: str = Field(default="$SCRATCH", description="Working directory path")
    requirements: str = Field(default="", description="venv requirements")
    use_apptainer_ubuntu: bool = Field(
        default=False, description="Use Apptainer Ubuntu"
    )
    need_pip_install: bool = Field(
        default=True,
        description="If this directory already has all packages installed, set this to False",
    )


def stdout2queue(session: Channel, q: queue.Queue, logger):
    """
    centralize stdout of a session
    """
    while not thread_stop_event.is_set():
        if session.recv_ready():
            data = session.recv(1024).decode("utf-8")
            data = data.replace("\r\n\r\n", "\n")
            data = clean_ansi_escape(data)
            q.put(data)
            logger.opt(raw=True).info(data + "\n")
            logfile_stream.write(data)
        time.sleep(0.1)


def stream_tunnel(host_name, local_port, remote_port, local_host, remote_host):
    retry_count = 0

    conn = None
    manager = None
    finished = None

    while not thread_stop_event.is_set():
        try:
            conn = Connection(host_name)
            conn.open()
            conn.transport.set_keepalive(30)
            logger.info("Tunnel SSH opened.")

            finished = threading.Event()
            manager = TunnelManager(
                local_port=local_port,
                local_host=local_host,
                remote_port=remote_port,
                remote_host=remote_host,
                transport=conn.transport,
                finished=finished,
            )
            manager.start()
            logger.info(
                f"Tunnel {local_host}:{local_port} -> {remote_host}:{remote_port}"
            )
            while not thread_stop_event.is_set():
                e = manager.exception()
                if e is None:
                    time.sleep(0.1)
                else:
                    e = e.value
                    logger.error(f"TunnelManager exception detected. {e}")
                    raise e
            logger.info("Tunnel manager closed")
        except Exception as e:
            logger.error(f"Tunnel Error: {e}\n{traceback.format_exc()}")
            logger.warning(f"Retrying ({retry_count})...")
            retry_count += 1
            time.sleep(1)
        except SystemExit:
            logger.warning("SystemExit detected.")
        finally:
            if finished:
                finished.set()
            if manager:
                manager.join()
            if conn:
                conn.close()

            conn = None
            manager = None
            finished = None

            logger.info("Tunnel SSH connection closed")
    return


def load_config():
    config_file = os.path.join(current_dir, "config.yaml")
    if not os.path.exists(config_file):
        logger.error("Configuration file config.yaml not found!")
        sys.exit(2)

    with open(config_file, "r") as f:
        yaml_str = yaml.safe_load(f)
    config = HPCServerConfig(**yaml_str)

    return config


def get_qrsh_cmd(config: HPCServerConfig):
    qrsh_template = "qrsh -N JUPYTER_SERVER -l {l_options} -pe {pe_options} -now n"
    l_options = ""
    pe_options = f"{config.parenv.value} {config.numberofslots}"

    if config.highp:
        l_options = "highp," + l_options

    l_options += f"h_rt={config.timeinhours}:00:00,h_data={config.memoryingb}G,arch={config.arch.value}"
    if config.usegpu:
        l_options += f",gpu,{config.gputype.value},cuda={config.cudanum}"
    if config.exclusive:
        l_options = l_options + "exclusive"

    qrsh_cmd = qrsh_template.format(l_options=l_options, pe_options=pe_options)
    return qrsh_cmd


def get_module_load_cmds(config: HPCServerConfig):
    out = list()
    # --- cuda
    if config.usegpu:
        out.append(f"module load {config.cudaver.value}")

    # --- additional modules
    if config.mods:
        mod_list = config.mods.split(",")
        for mod in mod_list:
            out.append(f"module load {mod}")
    if config.use_apptainer_ubuntu:
        out.append("module load apptainer")

    # --- python & gcc
    out.append(f"module load {config.gccver}")
    out.append(f"module load {config.pythonver.value}")

    return out


def get_env_activate_cmd(config: HPCServerConfig):
    # --- conda env
    cmd = ""
    if config.cve:
        cmd = f"conda activate {config.cve}"
    elif config.pve:
        cmd = f"source {config.pve}/bin/activate"
    return cmd


def check_uv_exists(session: Channel, q: queue.Queue):
    global uv_path
    session.send("echo UVPATH=`which uv`" + "\n")

    def _helper(line: str):
        global uv_path
        uv_path = ""
        splits = line.split("=")
        if splits[0] == "UVPATH":
            if len(splits) == 2 and splits[1].strip():
                uv_path = splits[1].strip()
            return True
        return False

    readwhile(q, _helper, timeout=10)
    result = uv_path
    if result:
        logger.info(f"uv exists at {result}")
        result = True
    else:  # try to install
        session.send("wget -qO- https://astral.sh/uv/install.sh | sh" + "\n")
        readwhile(q, _helper, timeout=10)
        if uv_path:
            logger.info("uv installed")
            result = True
        else:
            raise ValueError(
                "fail to install uv, try to install manually: wget -qO- https://astral.sh/uv/install.sh | sh"
            )
    del uv_path
    return result


def check_hostname(session: Channel, q: queue.Queue):
    """make sure don't start job on login node"""
    global remote_host

    def is_hostname(line: str):
        global remote_host
        if line.startswith("HOSTNAME"):
            remote_host = line.split("=")[1].strip()
            return True
        else:
            return False

    session.send("echo HOSTNAME=`hostname`\n")
    readwhile(q, is_hostname, timeout=30)

    if "login" in remote_host:
        logger.error(
            "You are be on the wrong host (login node rather than compute node). Please check your configuration and try again."
        )
        raise ValueError("should be compute node")
    logger.info(f"Connected to {remote_host}")
    return


def cd_create_venv(session: Channel, q: queue.Queue, config: HPCServerConfig):
    """
    Create a virtual environment and install dependencies
    """

    def _try_split_opt(pkg: str) -> tuple[list[str], str]:
        """
        --only-binary=:all: pyarrow
        -> options: [--only-binary=:all:]
        pkg_req: pyarrow
        """
        options = []
        pkg_splits = pkg.split(" ")
        pkg_req = pkg_splits[-1]
        for split in pkg_splits[:-1]:
            if split.startswith("-"):
                options.append(split)
        return options, pkg_req

    session.send("cd $SCRATCH" + "\n")
    session.send(f"mkdir {config.directory}" + "\n")
    session.send(f"cd {config.directory}" + "\n")
    if config.use_apptainer_ubuntu:
        session.send("apptainer run $H2_CONTAINER_LOC/ubuntu_22.04.sif" + "\n")
    time.sleep(2)
    session.send("uv venv --allow-existing -p 3.11" + "\n")
    time.sleep(0.5)
    if config.need_pip_install:
        pkgs = config.requirements.split("\n")
        pkgs.append("pyzmq<27")
        pkgs.append("jupyter_kernel_gateway")
        pkgs.append("ipykernel")
        pkgs = [pkg.strip() for pkg in pkgs if pkg.strip()]
        for pkg in pkgs:
            options, pkg_req = _try_split_opt(pkg)
            opt_str = " ".join(options)
            session.send(f"uv pip install {opt_str} '{pkg_req}'" + "\n")
            time.sleep(0.5)
        to = 1200
    else:
        logger.info("Assume packages already installed, skip installation")
        to = 5
    session.send("echo 'REQ_INSTALLED'\n")
    time.sleep(2)
    readwhile(q, lambda line: line.startswith("REQ_INSTALLED"), timeout=to)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True,
    retry=retry_if_exception(TimeoutError),
    before=lambda retry_state: logger.info(
        f"Attempting connection (attempt {retry_state.attempt_number})"
    ),
    after=lambda retry_state: logger.info(
        f"Connection attempt {retry_state.attempt_number} failed, used {retry_state.outcome_timestamp - retry_state.start_time:.2f}s since first retry"
    ),
)
def open_connection_with_retry(conn: Connection):
    """Open connection with retry logic using tenacity decorator."""
    try:
        conn.open()
        conn.transport.set_keepalive(30)
        logger.info("Connection established successfully")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping retry")
        raise


def get_available_port(session: Channel, q: queue.Queue) -> int:
    cmd = 'for port in {8789..65535}; do\n    (echo >/dev/tcp/127.0.0.1/$port) &>/dev/null || { echo "$port"; break; }\ndone\necho PORT=$port'
    session.send(cmd + "\n")

    def get_port(line: str):
        global avail_port
        if line.startswith("PORT"):
            avail_port = line.split("=")[1].strip()
            avail_port = int(avail_port)
            return True
        else:
            return False

    readwhile(q, get_port, timeout=60)
    logger.info(f"Available port: {avail_port}")
    return


def main():
    global remote_host
    global avail_port
    config: HPCServerConfig = load_config()

    conn = Connection(host=config.hostname)
    open_connection_with_retry(conn)

    session_qsub, q_qsub = get_session_queue(conn)

    thread_stdout_qsub = threading.Thread(
        target=stdout2queue, args=(session_qsub, q_qsub, qsub_stdout_logger)
    )
    thread_stdout_qsub.start()
    thread_tunnel = None

    try:
        # --- request compute node
        logger.info("requesting compute node...")
        start_time = time.time()

        qrsh_cmd = get_qrsh_cmd(config)
        logger.info(f"cmd to be executed: {qrsh_cmd}")

        session_qsub.send(qrsh_cmd + "\n")
        session_qsub.send("echo 'DONE'" + "\n")
        readwhile(q_qsub, lambda line: line.startswith("DONE"), timeout=None)
        logger.info(f"qrsh done, time used: {time.time() - start_time:.2f}")

        # --- check hostname
        check_hostname(session_qsub, q_qsub)

        # --- load modules
        module_load_cmds = get_module_load_cmds(config)
        for cmd in module_load_cmds:
            session_qsub.send(cmd + "\n")
        # show all modules
        session_qsub.send("module li" + "\n")

        # --- activate env
        env_activate_cmd = get_env_activate_cmd(config)
        if env_activate_cmd:
            session_qsub.send(env_activate_cmd + "\n")
        check_uv_exists(session_qsub, q_qsub)

        # --- open dir & install requirements
        cd_create_venv(session_qsub, q_qsub, config)

        # --- find available port
        get_available_port(session_qsub, q_qsub)

        # --- start jupyter kernel gateway
        session_qsub.send(
            f"uv run jupyter kernelgateway --KernelGatewayApp.ip=0.0.0.0 --KernelGatewayApp.port={avail_port}"
            + "\n"
        )
        time.sleep(5)

        # --- open tunnel
        local_host = "localhost"
        if config.usetunnel:
            logger.info("start tunnel thread...")
            thread_tunnel = threading.Thread(
                target=stream_tunnel,
                args=(
                    config.hostname,
                    config.port,
                    avail_port,
                    local_host,
                    remote_host,
                ),
            )
            thread_tunnel.start()

        # --- show count down
        for remaining in range(3600 * config.timeinhours, 0, -5):
            sys.stdout.write(f"\r⏳ Time left: {remaining:2d} seconds")
            sys.stdout.flush()
            time.sleep(5)

    except Exception as e:
        logger.error(f"Error: {e}, {traceback.format_exc()}")

    except SystemExit:
        logger.info("capture exit in main...")

    # --- close all sessions
    finally:
        # close all here
        thread_stop_event.set()
        thread_stdout_qsub.join()
        if thread_tunnel:
            thread_tunnel.join()
        session_qsub.close()
        conn.close()
        logfile_stream.close()
        logger.info("all connection closed.")
    return


if __name__ == "__main__":
    main()
