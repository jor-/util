import re


def get_int_in_string(string):
    return int(re.search(r'\d+', string).group())


def convert_format_string_in_regular_expression(format_string):
    regular_expression = format_string
    for pattern in re.findall('{.*:.*d}', regular_expression):
        regular_expression = regular_expression.replace(pattern, '[+,-]?[0-9]*')
    for pattern in re.findall('{.*}', regular_expression):
        regular_expression = regular_expression.replace(pattern, '.*')
    return regular_expression
