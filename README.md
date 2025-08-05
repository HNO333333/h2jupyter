## Usage

1. **Set up SSH access**
   1. Generate an SSH key pair on your local machine: `ssh-keygen -t rsa -b 4096`
   2. Copy your public key to the remote server's `~/.ssh/authorized_keys` file
   3. For detailed instructions, see: https://code.visualstudio.com/docs/remote/troubleshooting#_quick-start-using-ssh-keys

2. **Configure the application**
   1. Create a configuration file by copying the example: `cp config.yaml.example config.yaml`
   2. Edit `config.yaml` to match your environment:
      - Set `hostname` to your remote server's address (must match an entry in your SSH config)
      - Adjust resource requirements (time, memory, CPU architecture) as needed
      - Configure Python version and other modules to load
      - Set GPU options if required
      - Update the `directory` where the application will run `<directory>/h2jupyter` (defaults to `$SCRATCH`)
      - Add any additional Python packages to the `requirements` section

3. **Set up the Python environment**
   1. Create a virtual environment: `uv venv`
   2. Install dependencies: `uv sync`

4. **Run the server**
   1. Start the application: `uv run server.py`
   2. The server will request a compute node, set up the environment, and start a Jupyter kernel gateway

## What does it do

1. Request compute node using `qrsh` command with configured resource requirements (time, memory, CPU architecture, GPU if needed)
2. Open SSH session and interactive shell to the allocated compute node
3. Verify that we're on a compute node (not login node) by checking hostname
4. Navigate to configured working directory and create `h2jupyter` subdirectory
5. Load required modules (Python, GCC, CUDA if GPU is used) based on configuration
6. Check if `uv` package manager exists, install it if not present
7. Create virtual environment using `uv venv -p 3.11` in the h2jupyter directory
8. Install required packages including jupyter_kernel_gateway, ipykernel, and pyzmq<27 using `uv pip install`
9. Optionally open SSH tunnel in separate thread for secure communication between local and remote ports
10. Start Jupyter kernel gateway on the configured port (default 8789) with IP set to 0.0.0.0 to allow external connections