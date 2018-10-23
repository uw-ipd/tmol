def block_tensor_to_dense(block_tensor):
    """Convert [i, j, bi, bj] to [i * bi, j * bj] contiguous tensor.

    Converts a "blocked tensor", which may be sparse, into contiguous form.
    This is used to convert the results of block-sparse evaluations into dense
    pairwise results.

    This is a form of "reverse tiling" from 4-d [i, j, bi, bj]::

      i:
      0--- 1--- 2--- 3---

      bi:
      0123 0123 0123 0123 bi j

      aaaa bbbb cccc dddd 0  0
      xxxx xxxx xxxx xxxx 1  -

      eeee ffff gggg hhhh 0  1
      xxxx xxxx xxxx xxxx 2  -

    Is converted to the dense form [i * bi, j * bj]::


      012345..........

      aaaabbbbccccdddd 0
      xxxxxxxxxxxxxxxx 1
      eeeeffffgggghhhh 3
      xxxxxxxxxxxxxxxx 4

    """

    assert len(block_tensor.shape) == 4
    return (
        (block_tensor.to_dense() if block_tensor.is_sparse else block_tensor)
        .transpose(1, 2)
        .contiguous()
        .view(
            (
                block_tensor.shape[0] * block_tensor.shape[2],
                block_tensor.shape[1] * block_tensor.shape[3],
            )
        )
    )
