import os.path

def merge(strings, join_str='+', left_bracket='(', right_bracket=')'):
    commonprefix = os.path.commonprefix(strings)
    commonsuffix = os.path.commonprefix([string[::-1] for string in strings])[::-1]
    joined_string = join_str.join([string[len(commonprefix):len(string)-len(commonsuffix)] for string in strings])
    if len(commonprefix) > 0 or len(commonsuffix) > 0:
        joined_string = commonprefix + left_bracket + joined_string + right_bracket + commonsuffix
    return joined_string