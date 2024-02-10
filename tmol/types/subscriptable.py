import typing

class _SubscribedType(type):
    """
    This class is a placeholder to let the IDE know the attributes of the
    returned type after a __getitem__.
    """

    __origin__ = None
    __args__ = None

# adapted from typish (https://github.com/ramonhagenaars/typish)
class SubscriptableType(type):
    """
    This metaclass will allow a type to become subscriptable.

    >>> class SomeType(metaclass=SubscriptableType):
    ...     pass
    >>> SomeTypeSub = SomeType['some args']
    >>> SomeTypeSub.__args__
    'some args'
    >>> SomeTypeSub.__origin__.__name__
    'SomeType'
    """

    def __init_subclass__(mcs, **kwargs):
        mcs._hash = None
        mcs.__args__ = None
        mcs.__origin__ = None

    def __getitem__(self, item) -> _SubscribedType:
        body = {
            **self.__dict__,
            "__args__": item,
            "__origin__": self,
        }
        bases = self, *self.__bases__
        result = type(self.__name__, bases, body)
        if hasattr(result, "_after_subscription"):
            # TODO check if _after_subscription is static
            result._after_subscription(item)
        return result

    def __eq__(self, other):
        self_module = getattr(self, "__module__", None)
        self_qualname = getattr(self, "__qualname__", None)
        self_origin = getattr(self, "__origin__", None)
        self_args = getattr(self, "__args__", None)

        other_module = getattr(other, "__module__", None)
        other_qualname = getattr(other, "__qualname__", None)
        other_args = getattr(other, "__args__", None)
        other_origin = getattr(other, "__origin__", None)

        return (
            self_module == other_module
            and self_qualname == other_qualname
            and self_args == other_args
            and self_origin == other_origin
        )

    def __hash__(self):
        if not getattr(self, "_hash", None):
            self_module = getattr(self, "__module__", None)
            self_qualname = getattr(self, "__qualname__", None)
            self_origin = getattr(self, "__origin__", None)
            self_args = getattr(self, "__args__", None)
            self._hash = hash("{}{}{}{}".format(self_module, self_qualname, self_origin, self_args))
        return self._hash
