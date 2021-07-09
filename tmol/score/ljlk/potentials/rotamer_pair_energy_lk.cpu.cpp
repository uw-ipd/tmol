#include <tmol/score/ljlk/potentials/rotamer_pair_energy_lk.impl.hh>
#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

#include <tmol/pack/sim_anneal/compiled/annealer.hh>



namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
class LKRPECPUCalc : public pack::sim_anneal::compiled::RPECalc {
 public:
  LKRPECPUCalc(
    TView<Vec<Real, 3>, 3, D> context_coords,
    TView<Int, 2, D> context_block_type,
    TView<Vec<Real, 3>, 2, D> alternate_coords,
    TView<Vec<Int, 3>, 1, D> alternate_ids,
    TView<Int, 1, D> context_system_ids,
    TView<Int, 3, D> system_min_bond_separation,
    TView<Int, 5, D> system_inter_block_bondsep,
    TView<Int, 3, D> system_neighbor_list,
    TView<Int, 1, D> block_type_n_heavy_atoms,
    TView<Int, 2, D> block_type_heavyatom_index,
    TView<Int, 2, D> block_type_atom_types,
    TView<Int, 1, D> block_type_n_interblock_bonds,
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
    TView<Int, 3, D> block_type_path_distance,
    TView<LKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,
    TView<Real, 1, D> lj_lk_weights,
    TView<Real, 1, D> output
  ):
    context_coords_(context_coords),
    context_block_type_(context_block_type),
    alternate_coords_(alternate_coords),
    alternate_ids_(alternate_ids),
    context_system_ids_(context_system_ids),
    system_min_bond_separation_(system_min_bond_separation),
    system_inter_block_bondsep_(system_inter_block_bondsep),
    system_neighbor_list_(system_neighbor_list),
    block_type_n_heavy_atoms_(block_type_n_heavy_atoms),
    block_type_heavyatom_index_(block_type_heavyatom_index),
    block_type_atom_types_(block_type_atom_types),
    block_type_n_interblock_bonds_(block_type_n_interblock_bonds),
    block_type_atoms_forming_chemical_bonds_(block_type_atoms_forming_chemical_bonds),
    block_type_path_distance_(block_type_path_distance),
    type_params_(type_params),
    global_params_(global_params),
    lj_lk_weights_(lj_lk_weights),
    output_(output)
  {}

  void calc_energies() override {
    LKRPEDispatch<DeviceDispatch, D, Real, Int>::f(
      context_coords_,
      context_block_type_,
      alternate_coords_,
      alternate_ids_,
      context_system_ids_,
      system_min_bond_separation_,
      system_inter_block_bondsep_,
      system_neighbor_list_,
      block_type_n_heavy_atoms_,
      block_type_heavyatom_index_,
      block_type_atom_types_,
      block_type_n_interblock_bonds_,
      block_type_atoms_forming_chemical_bonds_,
      block_type_path_distance_,
      type_params_,
      global_params_,
      lj_lk_weights_,
      output_
    );
  }

  void finalize() override {}
  
 private:
  TView<Vec<Real, 3>, 3, D> context_coords_;
  TView<Int, 2, D> context_block_type_;
  TView<Vec<Real, 3>, 2, D> alternate_coords_;
  TView<Vec<Int, 3>, 1, D> alternate_ids_;
  TView<Int, 1, D> context_system_ids_;
  TView<Int, 3, D> system_min_bond_separation_;
  TView<Int, 5, D> system_inter_block_bondsep_;
  TView<Int, 3, D> system_neighbor_list_;
  TView<Int, 1, D> block_type_n_heavy_atoms_;
  TView<Int, 2, D> block_type_heavyatom_index_;
  TView<Int, 2, D> block_type_atom_types_;
  TView<Int, 1, D> block_type_n_interblock_bonds_;
  TView<Int, 2, D> block_type_atoms_forming_chemical_bonds_;
  TView<Int, 3, D> block_type_path_distance_;
  TView<LKTypeParams<Real>, 1, D> type_params_;
  TView<LJGlobalParams<Real>, 1, D> global_params_;
  TView<Real, 1, D> lj_lk_weights_;
  TView<Real, 1, D> output_;
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LKRPERegistratorDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 3, D> context_coords,
    TView<Int, 2, D> context_block_type,
    TView<Vec<Real, 3>, 2, D> alternate_coords,
    TView<Vec<Int, 3>, 1, D>
        alternate_ids,  // 0 == context id; 1 == block id; 2 == block type

    // which system does a given context belong to
    TView<Int, 1, D> context_system_ids,

    // dims: n-systems x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those blocks?
    // If this minimum is greater than the crossover, then no further logic for
    // deciding whether two atoms in those blocks should have their interaction
    // energies calculated: all should. intentionally small to (possibly) fit in
    // constant cache
    TView<Int, 3, D> system_min_bond_separation,

    // dims: n-systems x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> system_inter_block_bondsep,

    // dims n-systems x max-n-blocks x max-n-neighbors
    // -1 as the sentinel
    TView<Int, 3, D> system_neighbor_list,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_heavy_atoms,

    // index of the ith heavy atom in a block type
    TView<Int, 2, D> block_type_heavyatom_index,

    // what are the atom types for these atoms
    // Dimsize: n_block_types x max_n_atoms
    TView<Int, 2, D> block_type_atom_types,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_path_distance,
    //////////////////////

    // LJ parameters
    TView<LKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,
    TView<Real, 1, D> lj_lk_weights,
    TView<Real, 1, D> output,
    TView<int64_t, 1, tmol::Device::CPU> annealer) -> void {
  using tmol::pack::sim_anneal::compiled::RPECalc;
  using tmol::pack::sim_anneal::compiled::SimAnnealer;

  int64_t annealer_uint = annealer[0];
  SimAnnealer *sim_annealer = reinterpret_cast<SimAnnealer *>(annealer_uint);
  std::shared_ptr<RPECalc> calc =
      std::make_shared<LKRPECPUCalc<DeviceDispatch, D, Real, Int>>(
	context_coords,
	context_block_type,
	alternate_coords,
	alternate_ids,
	context_system_ids,
	system_min_bond_separation,
	system_inter_block_bondsep,
	system_neighbor_list,
	block_type_n_heavy_atoms,
	block_type_heavyatom_index,
	block_type_atom_types,
	block_type_n_interblock_bonds,
	block_type_atoms_forming_chemical_bonds,
	block_type_path_distance,
	type_params,
	global_params,
	lj_lk_weights,
	output
      );

  sim_annealer->add_score_component(calc);
}


template struct LKRPEDispatch<ForallDispatch, tmol::Device::CPU, float, int>;
template struct LKRPEDispatch<ForallDispatch, tmol::Device::CPU, double, int>;

template struct LKRPERegistratorDispatch<
    ForallDispatch,
    tmol::Device::CPU,
    float,
    int>;
template struct LKRPERegistratorDispatch<
    ForallDispatch,
    tmol::Device::CPU,
    double,
    int>;


}
}
}
}
