from __future__ import annotations

import argparse
from pathlib import Path

from stitch.engines.sglang import SGLangEngine
from stitch.service import serve
from stitch.types import VersionRef

from .fft_bulletin import (
    FFTSnapshotBulletin,
    FFTSnapshotStore,
    PinnedFFTSnapshotStore,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--base-checkpoint-dir", required=True)
    parser.add_argument("--bulletin-root", required=True)
    parser.add_argument("--bulletin-volume", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pinned-version", type=int)
    args = parser.parse_args()

    refresh = None
    if args.bulletin_volume:
        import modal

        volume = modal.Volume.from_name(args.bulletin_volume, version=2)
        refresh = volume.reload
    bulletin = FFTSnapshotBulletin(
        Path(args.bulletin_root),
        refresh=refresh,
    )
    store = FFTSnapshotStore(bulletin, args.run_id)
    if args.pinned_version is not None:
        store = PinnedFFTSnapshotStore(
            bulletin,
            VersionRef(args.run_id, args.pinned_version),
        )
    engine = SGLangEngine(
        args.upstream_url,
        args.base_checkpoint_dir,
        delta_update_mode="cpu",
    )
    serve(
        store,
        engine,
        run_id=args.run_id,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
