from functools import wraps


def print_if_complete(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'{func.__name__} starting')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('{} complete in {} seconds'.format(func.__name__, int(end-start)))
        return result
    return wrapper
