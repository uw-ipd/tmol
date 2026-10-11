from tmol_jax.score.common.acceptor import acceptor_bases


def test_current_native_acceptor_frames():
    def select(hybrid, graph, hydrogens=(), blocks=None):
        return acceptor_bases(
            0,
            hybrid,
            graph.__getitem__,
            lambda atom: atom in hydrogens,
            block=(lambda atom: blocks[atom]) if blocks else (lambda atom: 0),
        )

    graph = {0: [1, 2, 3], 1: [0, 4]}
    assert select(1, graph, [3]) == (1, 4)
    assert select(2, graph, [3]) == (1, 3)
    assert select(2, {0: [1, 2]}) == (1, 2)
    assert select(2, {0: [1]}) == (1, 1)
    assert select(3, graph, [3]) == (1, 2)
    assert select(3, {0: [1]}) == (-1, -1)
    assert select(1, {0: [1], 1: [0, 2, 3]}, blocks=[0, 1, 0, 1]) == (1, 3)
