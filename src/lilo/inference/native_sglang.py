"""SGLang entrypoint for resolved YAML serving options."""

import argparse
import json
import logging
import os
import sys

from lilo.argparse_config import apply_config_overrides


def main():
    from sglang.srt.server_args import ServerArgs
    from sglang.launch_server import run_server
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.plugins import load_plugins

    load_plugins()
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = ["--model-path", sys.argv[1], "--host", "127.0.0.1", "--port", "8001"]
    apply_config_overrides(parser, json.loads(sys.argv[2]), argv)
    raw = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, raw.log_level.upper()))
    args = ServerArgs.from_cli_args(raw)
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
