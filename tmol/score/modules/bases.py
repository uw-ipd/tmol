from abc import ABC, abstractmethod
import torch
import attr
from attrs_strict import type_validator
from typing import Dict, List, Type, Set, TypeVar, Union, Iterable
from collections import Counter, ChainMap
import itertools
from tmol.extern.toposort import toposort


@attr.s(auto_attribs=True)
class ScoreSystem:
    modules: Dict[Type["ScoreModule"], "ScoreModule"]
    methods: Dict[Type["ScoreMethod"], "ScoreMethod"]
    weights: Dict[str, float]

    @classmethod
    def build_for(
        cls,
        val,
        methods: Iterable[Type["ScoreMethod"]],
        weights: Dict[str, float],
        **kwargs,
    ):
        method_deps: Set[Type["ScoreModule"]] = {
            dep for m in methods for dep in m.depends_on()
        }

        instance = cls._build_with_modules(val, method_deps, **kwargs)
        instance.weights = weights
        instance.methods = {m: m.build_for(val, instance, **kwargs) for m in methods}

        return instance

    @classmethod
    def _build_with_modules(cls, val, modules: Iterable[Type["ScoreModule"]], **kwargs):
        module_and_deps: Dict[Type["ScoreModule"], Set[Type["ScoreModule"]]] = {}
        unresolved_modules: Set[Type["ScoreModule"]] = set(modules)

        while unresolved_modules:
            module = unresolved_modules.pop()

            assert issubclass(
                module, ScoreModule
            ), f"Resolved dep is not ScoreModule subclass: {module}"

            module_and_deps[module] = module.depends_on()

            for d in module_and_deps[module]:

                if d not in module_and_deps:
                    unresolved_modules.add(d)

        instance = ScoreSystem(dict(), dict(), dict())

        for module_generation in toposort(module_and_deps):
            instance.modules.update(
                {m: m.build_for(val, instance, **kwargs) for m in module_generation}
            )

        return instance

    def intra_forward(self, coords: torch.Tensor):

        terms: List[Dict[str, torch.Tensor]] = [
            method.intra_forward(coords) for method in self.methods.values()
        ]

        term_counts = Counter(itertools.chain(*terms))
        assert all(
            v == 1 for v in term_counts.values()
        ), f"Duplicate term counts: {dict(term_counts)}"

        return dict(ChainMap(*terms))

    def intra_total(self, coords: torch.Tensor):
        terms = self.intra_forward(coords)

        assert set(self.weights) == set(
            terms
        ), "Mismatched weights/terms: {self.weights} {terms}"

        return sum(self.weights[t] * v for t, v in terms.items())


_TModule = TypeVar("_TModule", bound="ScoreModule")


@attr.s(auto_attribs=True, slots=True, kw_only=True, frozen=True)
class ScoreModule(ABC):

    system: ScoreSystem = attr.ib(validator=type_validator())

    @classmethod
    def get(
        cls: Type[_TModule],
        system_or_module: Union[ScoreSystem, "ScoreMethod", "ScoreModule"],
    ) -> _TModule:

        system = (
            system_or_module
            if isinstance(system_or_module, ScoreSystem)
            else system_or_module.system
        )
        instance = system.modules[cls]
        assert isinstance(instance, cls)
        return instance

    @classmethod
    @abstractmethod
    def depends_on(cls: Type[_TModule]) -> Set[Type["ScoreModule"]]:
        raise NotImplementedError(f"ScoreModule.depends_on: {cls}")

    @classmethod
    @abstractmethod
    def build_for(
        cls: Type[_TModule], val: object, system: ScoreSystem, **_
    ) -> _TModule:
        raise NotImplementedError(f"ScoreModule.build_for: {cls}")


_TMethod = TypeVar("_TMethod", bound="ScoreMethod")


@attr.s(auto_attribs=True, slots=True, kw_only=True, frozen=True)
class ScoreMethod(ABC):

    system: ScoreSystem = attr.ib(validator=type_validator())

    @classmethod
    @abstractmethod
    def depends_on(cls: Type[_TMethod]) -> Set[Type["ScoreModule"]]:
        raise NotImplementedError("ScoreMethod.depends_on")

    @classmethod
    @abstractmethod
    def build_for(
        cls: Type[_TMethod], val: object, system: ScoreSystem, **_
    ) -> _TMethod:
        raise NotImplementedError("ScoreMethod.build_for")

    @abstractmethod
    def intra_forward(self, coords: torch.Tensor):
        raise NotImplementedError("intra_forward")
