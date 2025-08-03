import os
import queue
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Optional

import yaml
from fabric2 import Connection
from loguru import logger as base
from paramiko import Channel
from pydantic import BaseModel, Field
from tqdm import tqdm

# --- global vars
current_dir = os.path.dirname(os.path.abspath(__file__))
thread_stop_event = threading.Event()
remote_host = ""
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
                    logger.info(line)
                if condition_func(line):
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


def stdout2queue(session: Channel, q: queue.Queue, logger):
    """
    centralize stdout of a session
    """
    while not thread_stop_event.is_set():
        if session.recv_ready():
            data = session.recv(1024).decode("utf-8")
            data = data.replace("\r\n\r\n", "\n")
            q.put(data)
            logger.opt(raw=True).info(data + "\n")
            logfile_stream.write(data)
        time.sleep(1)


def stream_tunnel(host_name, local_port, remote_port, local_host, remote_host):
    try:
        conn = Connection(host_name)
        conn.open()
        logger.info(f"SSH connection to {host_name} established")
        with conn.forward_local(
            local_port=local_port, remote_host=remote_host, remote_port=remote_port
        ):
            logger.info(
                f"Port forwarding established: {local_host}:{local_port} -> {remote_host}:{remote_port}"
            )
            while not thread_stop_event.is_set():
                time.sleep(0.2)
        logger.info("Port forwarding closed")
    except Exception as e:
        logger.info(f"Tunnel Error: {e}")
        raise e
    finally:
        conn.close()
        logger.info("Tunnel connection closed")
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
    if config.exclusive:
        l_options = "exclusive," + l_options

    l_options = f"h_rt={config.timeinhours}:00:00,h_data={config.memoryingb}G,arch={config.arch.value}"
    if config.usegpu:
        l_options += f",gpu,{config.gputype.value},cuda={config.cudanum}"

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
    session.send(f"cd {config.directory}\n")
    session.send("mkdir h2jupyter" + "\n")
    session.send("cd h2jupyter" + "\n")
    session.send("uv venv -p 3.11" + "\n")
    pkgs = config.requirements.split("\n")
    pkgs.append("pyzmq<27")
    pkgs.append("jupyter_kernel_gateway")
    pkgs.append("ipykernel")
    pkgs = [pkg.strip() for pkg in pkgs if pkg.strip()]
    for pkg in pkgs:
        session.send(f"uv pip install '{pkg}'\n")
    session.send("echo REQ_INSTALLED\n")
    readwhile(q, lambda line: line.startswith("REQ_INSTALLED"), timeout=1200)


def main():
    global remote_host
    config: HPCServerConfig = load_config()

    conn = Connection(host=config.hostname)
    conn.open()

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

        # --- open dir
        session_qsub.send(f"cd {config.directory}" + "\n")
        session_qsub.send(f"echo '{get_current_time_str()}' >> temp.txt" + "\n")

        # --- open dir & install requirements
        cd_create_venv(session_qsub, q_qsub, config)

        # --- start jupyter kernel gateway
        session_qsub.send(
            f"uv run jupyter kernelgateway --KernelGatewayApp.ip=0.0.0.0 --KernelGatewayApp.port={config.port}"
            + "\n"
        )

        # --- open tunnel
        local_host = "localhost"
        thread_tunnel = threading.Thread(
            target=stream_tunnel,
            args=(
                config.hostname,
                config.port,
                config.port,
                local_host,
                remote_host,
            ),
        )
        thread_tunnel.start()

        # --- show count down
        start_time = time.time()
        for _ in tqdm(range(3600 * config.timeinhours, 0, -30)):
            time.sleep(30)

    except KeyboardInterrupt:
        logger.error("User interrupted.")

    except Exception as e:
        logger.error(f"Error: {e}, {str(e.__traceback__)}")

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


def test():
    config = load_config()

    conn = Connection(config.hostname)
    conn.open()
    session = conn.transport.open_session()
    session.get_pty()
    session.invoke_shell()

    q = queue.Queue()
    stream_thread = threading.Thread(
        target=stdout2queue,
        args=(
            session,
            q,
            qsub_stdout_logger,
        ),
    )
    stream_thread.start()

    try:
        for _ in range(5):
            session.send("echo 'hello world!'" + "\n")
            time.sleep(0.1)
        session.send("echo 'last world!'" + "\n")
        readwhile(q, lambda line: line.startswith("last"), verbose=False)
        logger.info("+" * 30)
        for _ in range(5):
            session.send("echo 'hihi'" + "\n")
            time.sleep(0.1)
        session.send("echo 'last hi!'" + "\n")
        readwhile(q, lambda line: line.startswith("last"), verbose=False)
        logger.info("=" * 30)
    except Exception as e:
        logger.error(f"Error: {str(e)}. Traceback:\n{e.__traceback__}")
    finally:
        thread_stop_event.set()
        stream_thread.join()
        session.close()
        conn.close()
        logfile_stream.close()
        logger.info("connection closed.")


if __name__ == "__main__":
    main()
