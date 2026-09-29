"""Content-free transport test only. Never launches an acquisition."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hplc_al.common import ROOT, atomic_json
from hplc_al.llm.credentials import inject_key
from hplc_al.llm.execution import preflight_with_retries
from hplc_al.llm.responses_transport import TransportError, preflight, settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args(argv)
    path = (
        ROOT
        / "studies/active_learning/odh_free_llm32_fullpool_v2/transport_preflight.json"
    )
    # Launcher injects only the user-designated key; transport remains env-only.
    try:
        available = inject_key(path=args.key_file)
    except TransportError as error:
        print(str(error), file=sys.stderr)
        return 1
    if not available:
        atomic_json(
            path,
            {
                "status": "READY_FOR_API_KEY",
                "error": "RESPONSES_API_KEY_NOT_CONFIGURED",
                "missing_environment_variable": "TOKEN4RESEARCH_API_KEY",
                "requests_sent": 0,
                "scientific_content": False,
            },
        )
        print(
            "RESPONSES_API_KEY_NOT_CONFIGURED: TOKEN4RESEARCH_API_KEY\nREADY_FOR_API_KEY"
        )
        return 0
    try:
        config = settings()
        from hplc_al.llm.runner import EXPECTED_CONFIG

        if config != EXPECTED_CONFIG:
            raise TransportError(
                "RESPONSES_CONFIG_MISMATCH: configure token4research / gpt-6-astra / high as registered"
            )
        preflight_with_retries(lambda: preflight(path, config))
    except TransportError as error:
        print(str(error), file=sys.stderr)
        return 1
    print("READY_FOR_SCIENTIFIC_RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
