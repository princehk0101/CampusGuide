import argparse
import sys

from backend.university_sources import (
    UNIVERSITY_SOURCES,
    refresh_linked_university_documents,
    refresh_university_sources,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Refresh official Bundelkhand University source files."
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=[source.key for source in UNIVERSITY_SOURCES],
        help=(
            "Refresh only this source key; may be repeated. Without this "
            "option, refresh the complete source pack."
        ),
    )
    parser.add_argument(
        "--linked",
        action="store_true",
        help=(
            "Also refresh every syllabus and examination PDF linked from "
            "the official BU index pages."
        ),
    )
    args = parser.parse_args()

    results = refresh_university_sources(
        source_keys=set(args.source) if args.source else None
    )
    if not args.source or args.linked:
        results.extend(refresh_linked_university_documents())
    failures = 0
    for result in results:
        print(f"{result['status']}: {result['title']}")
        if result["status"] not in {"updated", "current"}:
            failures += 1

    updated_count = sum(
        result["status"] in {"updated", "current"}
        for result in results
    )
    print(
        f"Available {updated_count} of {len(results)} selected Bundelkhand "
        "University sources."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
