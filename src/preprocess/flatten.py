def flatten_dict(data: dict) -> str:
    """
    Flatten a dictionary into a string.
    """
    output = ''
    for key, value in data.items():
        if isinstance(value, dict):
            output += flatten_dict(value)
        else:
            output += f'{key}: {value}; '
        output = output.strip()
        output += '\n'
    
    return output.strip()