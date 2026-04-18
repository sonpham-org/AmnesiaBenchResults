# Author: Claude Sonnet 4.6 (Bubba)
# Date: 29-March-2026
# PURPOSE: Top-level runner for AmnesiaBench v3. Thin wrapper — imports and calls the
#   package CLI. Run as: python3 run_bench.py <subcommand> [args].
#   Integration points: delegates entirely to amnesia_bench.cli.main().
# SRP/DRY check: Pass — zero business logic here; all routing is in cli.py.

from amnesia_bench.cli import main

if __name__ == "__main__":
    main()
