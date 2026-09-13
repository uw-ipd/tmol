"""One current parameter annotation per object, with weak source identities."""

from dataclasses import dataclass
import weakref


@dataclass(frozen=True, slots=True, eq=False)
class AnnotationKey:
    sources: tuple
    settings: tuple = ()

    @classmethod
    def from_sources(cls, *sources, settings=()):
        return cls(tuple(weakref.ref(source) for source in sources), settings)

    def matches(self, other):
        # A compiled configuration remains valid after its source expires.
        # Distinct keys still require live, identical source referents.
        if self is other:
            return True
        return (
            isinstance(other, AnnotationKey)
            and self.settings == other.settings
            and len(self.sources) == len(other.sources)
            and all(
                (source := first()) is not None and source is second()
                for first, second in zip(self.sources, other.sources)
            )
        )


def cached_annotation(owner, attribute, key):
    entry = getattr(owner, attribute, None)
    return entry[1] if entry is not None and key.matches(entry[0]) else None


def store_annotation(owner, attribute, key, value):
    setattr(owner, attribute, (key, value))
    return value
