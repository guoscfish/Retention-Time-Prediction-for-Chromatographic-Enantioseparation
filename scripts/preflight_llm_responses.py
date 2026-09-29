"""Content-free transport test only. Never launches an acquisition."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hplc_al.common import ROOT, atomic_json
from hplc_al.llm.responses_transport import TransportError, preflight, settings


def main():
    path = (
        ROOT
        / "studies/active_learning/odh_free_llm32_fullpool_v2/transport_preflight.json"
    )
    # Explicit task target; no loading of credentials or fallback authentication.
    if not os.environ.get("TOKEN4RESEARCH_API_KEY"):
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
        preflight(path, config)
    except TransportError as error:
        print(str(error), file=sys.stderr)
        return 1
    print("READY_FOR_SCIENTIFIC_RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
