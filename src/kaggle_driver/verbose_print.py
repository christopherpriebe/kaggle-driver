"""A module for formatting and printing messages to the console.
"""
def print_title(title: str) -> None:
    """Prints a title.

    :param title: The title to print.
    :type title: str
    """
    print(f"\n{title}\n{'-' * len(title)}")


def print_msg(msg: str) -> None:
    """Prints a message.

    :param msg: The message to print.
    :type msg: str
    """
    formatted_msg: str = "| " + msg.replace("\n", "\n| ")
    if formatted_msg.endswith("\n| "):
        formatted_msg = formatted_msg[:-3]
    print(formatted_msg)
