import datetime

def datetime_to_float(dt):
    year_start = datetime.datetime(dt.year, 1, 1)
    next_year_start = datetime.datetime(dt.year + 1, 1, 1)
    float_fraction = (dt - year_start).total_seconds() / ((next_year_start - year_start).total_seconds())

    return dt.year + float_fraction