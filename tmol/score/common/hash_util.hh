#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <tmol/score/common/diamond_macros.hh>

namespace tmol {
namespace score {
namespace common {

// This hash function must match the hash function that was used to generate the
// key table
template <typename Int, Int key_size>
TMOL_DEVICE_FUNC int hash_funct(Vec<Int, key_size> key, int max_size) {
  int value = 0x1234;
  for (int i = 0; i < key_size; i++) {
    int k = key[i];
    if (k == -1) break;
    value = (k ^ value) * 3141 % max_size;
  }
  return value;
}

// Given a key (Vec of Ints), and a key table, give the index of the entry in
// the value table Return -1 if the key doesn't exist. The key table must use
// linear probing for collision resolution.
template <typename Int, Int key_size, tmol::Device D>
TMOL_DEVICE_FUNC int hash_lookup(
    Vec<Int, key_size> key, TView<Vec<Int, key_size + 1>, 1, D> hash_keys) {
  int index = hash_funct<Int>(key, hash_keys.size(0));
  while (true) {
    bool match = true;
    for (int i = 0; i < key_size; i++) {
      match = match && key[i] == hash_keys[index][i];
    }
    if (match) return hash_keys[index][key_size];

    if (hash_keys[index][0] == -1) return -1;

    index++;
    index = index % hash_keys.size(0);
  }
}

}  // namespace common
}  // namespace score
}  // namespace tmol
