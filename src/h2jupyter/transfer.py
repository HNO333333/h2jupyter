from pathlib import Path

from fabric2 import Connection
from scp import SCPClient
from loguru import logger


def upload(
    local_path: str,
    username: str,
    remote_base_dir: str = "$SCRATCH/h2jupyter",
    hostname="hoffman2-dtn",
):
    """
    Upload a file or directory to a remote host using SCP.

    Args:
        local_path (str): Path to the local file or directory to upload
        remote_base_dir (str): Base directory on the remote host where the file/directory will be uploaded
        hostname (str): Hostname of the remote server (default: hoffman2-dtn)
    """
    if "$SCRATCH" in remote_base_dir:
        remote_base_dir = remote_base_dir.replace(
            "$SCRATCH", f"/u/scratch/{username[0]}/{username}"
        )

    # Convert to Path object for better cross-platform handling
    local_path_obj = Path(local_path)

    # Check if local_path exists
    if not local_path_obj.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    # Determine if local_path is a file or directory
    is_directory = local_path_obj.is_dir()

    # Extract the base name of the local path
    local_name = local_path_obj.name

    # Construct the full remote path using forward slashes for Unix compatibility
    remote_path = f"{remote_base_dir.rstrip('/')}/{local_name}"

    # Establish connection
    ssh = Connection(hostname)
    ssh.open()

    try:
        logger.info(f"transfer from {str(local_path_obj)} to {remote_path}")
        # Use SCP to transfer file or directory
        with SCPClient(ssh.client.get_transport()) as scp:
            scp.put(str(local_path_obj), remote_path, recursive=is_directory)
    finally:
        # Ensure connection is closed
        ssh.close()
    return

def download(
    remote_rel_path: str,
    local_base_dir: str,
    username: str,
    remote_base_dir: str = "$SCRATCH/h2jupyter",
    hostname="hoffman2-dtn",
):
    """
    Download a file or directory from a remote host using SCP.

    Args:
        remote_path (str): Path to the remote file or directory to download
        local_base_dir (str): Local directory where the file/directory will be downloaded
        username (str): Username for the remote host
        remote_base_dir (str): Base directory on the remote host (default: $SCRATCH/h2jupyter)
        hostname (str): Hostname of the remote server (default: hoffman2-dtn)
    """
    if "$SCRATCH" in remote_base_dir:
        remote_base_dir = remote_base_dir.replace(
            "$SCRATCH", f"/u/scratch/{username[0]}/{username}"
        )

    # Convert to Path object for better cross-platform handling
    local_base_dir_obj = Path(local_base_dir)

    # Check if local_base_dir exists
    if not local_base_dir_obj.exists():
        raise FileNotFoundError(f"Local base directory does not exist: {local_base_dir}")

    # Ensure local_base_dir is a directory
    if not local_base_dir_obj.is_dir():
        raise NotADirectoryError(f"Local base path is not a directory: {local_base_dir}")

    # Extract the base name of the remote path
    remote_name = Path(remote_rel_path).name

    # Construct the full local path using forward slashes for Unix compatibility
    local_path = f"{local_base_dir.rstrip('/')}/{remote_name}"
    
    remote_path = f"{remote_base_dir.rstrip('/')}/{remote_name}"

    # Establish connection
    ssh = Connection(hostname)
    ssh.open()

    try:
        logger.info(f"transfer from {remote_path} to {local_path}")
        # Use SCP to transfer file or directory
        with SCPClient(ssh.client.get_transport()) as scp:
            scp.get(remote_path, local_path, recursive=True)
    finally:
        # Ensure connection is closed
        ssh.close()
    return


# Example usage:
# upload('./timeseriesgym', '/u/scratch/s/shuhan/h2jupyter')
# This will upload the local 'timeseriesgym' directory to '/u/scratch/s/shuhan/h2jupyter/timeseriesgym'
# download('/u/scratch/s/shuhan/h2jupyter/timeseriesgym', './timeseriesgym')
# This will download the remote '/u/scratch/s/shuhan/h2jupyter/timeseriesgym' directory to the local 'timeseriesgym' directory