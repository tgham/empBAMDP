import random
import string


def generate_random_id(length: int) -> str:
    """
    Generate random participant id

    :param length: The length of the id to generate
    :return: A new random id length characters long
    """
    return ''.join(
        random.choices(
            string.ascii_lowercase + string.digits,
            k=length
        )
    )
