from functools import wraps


def print_if_complete(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'{func.__name__} complete')
        return func(*args, **kwargs)
    return wrapper
