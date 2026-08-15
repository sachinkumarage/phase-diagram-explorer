import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase-diagram-explorer",
        description="Explore and visualize thermodynamic phase diagrams",
    )
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    print("Phase Diagram Explorer")


if __name__ == "__main__":
    main()
