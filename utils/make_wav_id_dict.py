def make_wav_id_dict(file_list):
    """
    Args:
        file_list(List[str]): List of DNS challenge filenames.

    Returns:
        dict: Look like {file_id: filename, ...}
    """
    return {get_file_id(fp): fp for fp in file_list}

def get_file_id(fp):
    """Split string to get wave id in DNS challenge dataset."""
    return fp.split("_")[-1].split(".")[0]