import logging


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
