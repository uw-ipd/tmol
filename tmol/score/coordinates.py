import properties
from tmol.properties.array import VariableT


class RealSpaceScoreGraph(properties.HasProperties):
    coords = VariableT("source atomic coordinates")

    def step(self):
        """Recalculate total_score and gradients wrt/ coords. Does not clear coord grads."""

        self._notify(
            dict(
                name="coords",
                prev=getattr(self, "coords"),
                mode="observe_set"
            )
        )

        self.total_score.backward()
        return self.total_score
