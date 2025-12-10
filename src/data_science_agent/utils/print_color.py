from enums import Color


def print_color(text: str, color: Color) -> None:
    """Prints the given text in the specified color."""
    print(f"{color}{text}{Color.ENDC}")
