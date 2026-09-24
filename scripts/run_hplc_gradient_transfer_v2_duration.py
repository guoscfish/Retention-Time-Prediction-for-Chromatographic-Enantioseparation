"""Run only the fixed-L333 two-seed duration smoke; no AL acquisitions."""

import json

from hplc_al.transfer_v2_duration import run_duration_smoke


if __name__ == "__main__":
    print(json.dumps(run_duration_smoke(), indent=2))
