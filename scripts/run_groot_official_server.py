"""Run the official GR00T N1.7 policy server from the complete local checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from groot_runtime import load_official_libero_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/workspace/assets/groot_n17/repo/libero_10"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5559)
    args = parser.parse_args()
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    policy = Gr00tSimPolicyWrapper(load_official_libero_policy(args.checkpoint))
    with PolicyServer(policy=policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
