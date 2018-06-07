import tmol.utility.mixins as mixins


class Factory:
    @classmethod
    def build_for(cls, val, **kwargs):
        return cls(**cls.init_parameters_for(val, **kwargs))

    @classmethod
    def init_parameters_for(cls, val, **kwargs):
        return mixins.cooperative_superclass_factory(
            cls,
            "factory_for",
            val,
            **kwargs,
        )
