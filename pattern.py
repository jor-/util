import re


def is_matching(string, pattern, integer_pattern='{:0>\d+}'):
    for integer_pattern_i in re.findall(integer_pattern, pattern):
            pattern = pattern.replace(integer_pattern_i, '\d+')
    find_all = re.findall(pattern, string)
    is_matching = len(find_all) == 1 and string == find_all[0]

    return is_matching


def get_int_in_string(string):
    return int(re.search(r'\d+', string).group())