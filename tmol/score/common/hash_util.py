import numpy


# Utility functions for constructing a hash table from a tuple of ints to a tuple of floats
# The implementation of the hash function here must match the corresponding function in
# hash_util.hh in this same directory.


def make_hashtable_keys_values(total_size, scale_factor, key_len, value_len):
    hash_keys = numpy.full(
        (total_size * scale_factor, key_len),
        -1,
        dtype=numpy.int32,
    )
    hash_values = numpy.full(
        (total_size, value_len),
        0,
        dtype=numpy.float32,
    )

    return hash_keys, hash_values


def hash_fun(key, max_size):
    value = 0x1234
    for k in key:
        value = (k ^ value) * 3141 % max_size  # XOR
    return value


def add_to_hashtable(hash_keys, hash_values, cur_value_index, key, values):
    index = hash_fun(key, hash_keys.shape[0])
    while hash_keys[index][0] != -1:
        index = (index + 1) % hash_keys.shape[0]
    for i, k in enumerate(key):
        hash_keys[index][i] = k
    hash_keys[index][4] = cur_value_index

    for i, value in enumerate(values):
        hash_values[cur_value_index][i] = value
