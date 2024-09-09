import json


def parse_json(filename: str) -> dict:
    """Parses data from 'filename' into a dictionary.

    Note: This function was copied from the template of exercise 1.

    Parameters:
    -----------
    filename : str
        The path to the of a .json file.

    Returns:
    --------
    dict
        The dictionary with parsed key-value data.
    """

    with open(filename) as fin:
        content = json.load(fin)
    return content


def write_json(filename: str, content: dict):
    """Writes data from a dictionary into 'filename' .

    Parameters:
    -----------
    filename : str
        The path to the of a .json file.
    content : dict
        The dictionary that will be converted to a .json file.
    """

    with open(filename, 'w') as fin:
        json.dump(content, fin)
