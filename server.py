import os
import random
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Optional

import yaml
from fabric2 import Connection
from pydantic import BaseModel, Field

from utils import file_logger, get_current_time_str, logger

random.seed(datetime.now().strftime("%S"))

if sys.version_info < (2, 6, 0):
    sys.stderr.write("You need python 2.6 or later to run this script\n")
    exit(1)


def readwhile(session, condition_func, timeout=10):
    """
    Read from a stream until a condition is met or a timeout occurs.
    """
    start = time.time()

    while True:
        flag = False
        if session.recv_ready():
            output = session.recv(4096).decode("utf-8")
            for line in output.split("\n"):
                file_logger.info(output)
                if condition_func(line):
                    flag = True
                    break
        elif timeout:  # set timeout=None to disable timeout
            if time.time() - start > timeout:
                raise TimeoutError("No output received for too long.")
        else:
            time.sleep(0.1)
        if flag:
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

    def model_post_init(self, context):
        pass


def usage():
    print("Usage:")
    print("Create a server.yaml file with the required configuration.")
    print("See sample configuration below:")
    print("""
username: shuhan
timeinhours: 1
memoryingb: 4
parenv: shared # or dc\\*
numberofslots: 8 # number of computing cores
port: 8789
directory: $SCRATCH
pythonver: python/3.11.12 # can be 
usegpu: false
gpu: null
gpumem: null
cudaver: "11.3"
highp: false
exclusive: false
arch: null
mods: null
cve: null
pve: null
sshlogs: false
gccver: gcc/11.3.0
modules: null
""")
    sys.exit(2)


def load_config():
    config_file = "./config.yaml"
    if not os.path.exists(config_file):
        print("Configuration file server.yaml not found!")
        usage()
        sys.exit(2)

    with open(config_file, "r") as f:
        yaml_str = yaml.safe_load(f)
    config = HPCServerConfig(**yaml_str)

    return config


def get_ssh_args(config: HPCServerConfig):
    if not config.hostname:
        dst = f"{config.username}@hoffman2.idre.ucla.edu"
    else:
        dst = config.hostname

    ssh_args = [
        "ssh",
        "-o",
        "ServerAliveCountMax=5",
        "-o",
        "IPQoS=throughput",
        "-o",
        "ServerAliveInterval=30",
        "-X",
        "-Y",
        "-t",
        "-t",
        "-4",
    ]
    if config.sshlogs:
        ssh_args.extend(["-E", "myssh.log"])

    ssh_args.append(dst)  # at the end
    return ssh_args


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


def main():
    config = load_config()

    with Connection(config.hostname) as conn:
        conn.open()
        session_qsub = conn.transport.open_session()
        session_qsub.get_pty()
        session_qsub.invoke_shell()
        logger.info("login!")

        # --- request compute node
        qrsh_cmd = get_qrsh_cmd(config)
        logger.info(f"qrsh_cmd: {qrsh_cmd}")
        session_qsub.send(qrsh_cmd + "\n")
        session_qsub.send("echo 'DONE'" + "\n")
        readwhile(session_qsub, lambda line: line.startswith("DONE"), timeout=None)
        logger.info("qrsh done!")

        # load modules
        # --- load modules
        module_load_cmds = get_module_load_cmds(config)
        for cmd in module_load_cmds:
            session_qsub.send(cmd + "\n")

        # --- activate env
        env_activate_cmd = get_env_activate_cmd(config)
        if env_activate_cmd:
            session_qsub.send(env_activate_cmd + "\n")

        # --- open dir
        session_qsub.send(f"cd {config.directory}" + "\n")
        session_qsub.send(f"echo '{get_current_time_str()}' >> temp.txt" + "\n")

        # --- get hostname
        def is_hostname(line: str):
            global hostname
            if line.startswith("HOSTNAME"):
                hostname = line.split("=")[1].strip()
                return True
            else:
                return False

        session_qsub.send("echo HOSTNAME=`hostname`\n")
        readwhile(session_qsub, is_hostname, timeout=30)

        # --- close all sessions
        session_qsub.close()
    logger.info("connection closed.")
    return


if __name__ == "__main__":
    main()
