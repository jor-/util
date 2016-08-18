import sys

def multiprocessing_module():
    try:
        # globals()['multiprocessing']
        sys.modules['multiprocessing']
        return True
    except KeyError:
        return False


def scoop_module():
    try:
        import scoop
        return scoop.IS_RUNNING
    except ImportError:
        return False
