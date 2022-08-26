""" Exceptions and errors for edge classification """


class NoEdgesFoundError(Exception):
    """
    Exception raised when no edges are found in image
    """

    def __init__(self, message: str = "No edges detected."):
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message
