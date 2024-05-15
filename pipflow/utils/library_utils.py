import inspect

def _isproperty(object):
    """Return true if the object is a method descriptor."""
    if isinstance(object, property):
        return True

    if (
        callable(object)
        or inspect.isclass(object)
        or inspect.ismethod(object)
        or inspect.isfunction(object)
        or inspect.iscoroutine(object)
        or inspect.isgenerator(object)
    ):
        # mutual exclusion
        return False
    tp = type(object)
    return hasattr(tp, "__get__")