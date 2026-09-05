"""Installed ``apiro`` console entry point."""


def main() -> None:
    """Delegate directly to the packaged canonical CLI implementation."""
    from apiro.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
