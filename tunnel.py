import argparse
import os
import signal
import sys
import threading
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server import exit_handler, load_config, logger, stream_tunnel, thread_stop_event

signal.signal(signal.SIGINT, exit_handler)


def main():
    parser = argparse.ArgumentParser(description="Tunnel configuration")
    parser.add_argument("--remote-host", type=str, help="Remote host address")
    args = parser.parse_args()

    config = load_config()
    local_host = "localhost"
    remote_host = args.remote_host
    if not remote_host:
        logger.error("remote host is not specified")
        exit(0)
    logger.info("start tunnel thread...")
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
    try:
        while True:
            time.sleep(1)
    except SystemExit:
        pass
    finally:
        thread_stop_event.set()
        thread_tunnel.join()
    return


if __name__ == "__main__":
    main()
