#include <tmol/score/ljlk/potentials/rotamer_pair_energy_lj.impl.hh>
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
class LJLKRPECPUCalc : public pack::sim_anneal::compiled::RPECalc {
 public:
  LJLKRPECPUCalc(
      TView<Vec<Real, 3>, 2, D> context_coords,
      TView<Int, 2, D> context_coord_offsets,
      TView<Int, 2, D> context_block_type,
      TView<Vec<Real, 3>, 1, D> alternate_coords,
      TView<Int, 1, D> alternate_coord_offsets,
      TView<Vec<Int, 3>, 1, D>
          alternate_ids,  // 0 == context id; 1 == block id; 2 == block type

      // which system does a given context belong to
      TView<Int, 1, D> context_system_ids,

      // dims: n-systems x max-n-blocks x max-n-blocks
      // Quick lookup: given the inds of two blocks, ask: what is the minimum
      // number of chemical bonds that separate any pair of atoms in those
      // blocks? If this minimum is greater than the crossover, then no further
      // logic for deciding whether two atoms in those blocks should have their
      // interaction energies calculated: all should. intentionally small to
      // (possibly) fit in constant cache
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
      TView<Int, 1, D> block_type_n_atoms,
      TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
      TView<Int, 2, D> block_type_heavy_atoms_in_tile,

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
      TView<LJLKTypeParams<Real>, 1, D> type_params,
      TView<LJGlobalParams<Real>, 1, D> global_params,

      TView<Real, 1, D> lj_lk_weights,
      TView<Real, 1, D> output,

      TView<int64_t, 1, tmol::Device::CPU> score_event,
      TView<int64_t, 1, tmol::Device::CPU> annealer_event)
      : context_coords_(context_coords),
        context_coord_offsets_(context_coord_offsets),
        context_block_type_(context_block_type),
        alternate_coords_(alternate_coords),
        alternate_coord_offsets_(alternate_coord_offsets),
        alternate_ids_(alternate_ids),
        context_system_ids_(context_system_ids),
        system_min_bond_separation_(system_min_bond_separation),
        system_inter_block_bondsep_(system_inter_block_bondsep),
        system_neighbor_list_(system_neighbor_list),
        block_type_n_atoms_(block_type_n_atoms),
        block_type_n_heavy_atoms_in_tile_(block_type_n_heavy_atoms_in_tile),
        block_type_heavy_atoms_in_tile_(block_type_heavy_atoms_in_tile),
        block_type_atom_types_(block_type_atom_types),
        block_type_n_interblock_bonds_(block_type_n_interblock_bonds),
        block_type_atoms_forming_chemical_bonds_(
            block_type_atoms_forming_chemical_bonds),
        block_type_path_distance_(block_type_path_distance),
        type_params_(type_params),
        global_params_(global_params),
        lj_lk_weights_(lj_lk_weights),
        output_(output),
        score_event_(score_event),
        annealer_event_(annealer_event) {}

  void calc_energies() override {
    LJLKRPEDispatch<DeviceDispatch, D, Real, Int>::f(
        context_coords_,
        context_coord_offsets_,
        context_block_type_,
        alternate_coords_,
        alternate_coord_offsets_,
        alternate_ids_,
        context_system_ids_,
        system_min_bond_separation_,
        system_inter_block_bondsep_,
        system_neighbor_list_,
        block_type_n_atoms_,
        block_type_n_heavy_atoms_in_tile_,
        block_type_heavy_atoms_in_tile_,
        block_type_atom_types_,
        block_type_n_interblock_bonds_,
        block_type_atoms_forming_chemical_bonds_,
        block_type_path_distance_,
        type_params_,
        global_params_,
        lj_lk_weights_,
        output_,
        score_event_,
        annealer_event_);
  }

  void finalize() override {}

 private:
  TView<Vec<Real, 3>, 2, D> context_coords_;
  TView<Int, 2, D> context_coord_offsets_;
  TView<Int, 2, D> context_block_type_;
  TView<Vec<Real, 3>, 1, D> alternate_coords_;
  TView<Int, 1, D> alternate_coord_offsets_;
  TView<Vec<Int, 3>, 1, D> alternate_ids_;

  TView<Int, 1, D> context_system_ids_;
  TView<Int, 3, D> system_min_bond_separation_;

  TView<Int, 5, D> system_inter_block_bondsep_;

  TView<Int, 3, D> system_neighbor_list_;

  TView<Int, 1, D> block_type_n_atoms_;
  TView<Int, 2, D> block_type_n_heavy_atoms_in_tile_;
  TView<Int, 2, D> block_type_heavy_atoms_in_tile_;

  TView<Int, 2, D> block_type_atom_types_;

  TView<Int, 1, D> block_type_n_interblock_bonds_;

  TView<Int, 2, D> block_type_atoms_forming_chemical_bonds_;

  TView<Int, 3, D> block_type_path_distance_;

  // LJ parameters
  TView<LJLKTypeParams<Real>, 1, D> type_params_;
  TView<LJGlobalParams<Real>, 1, D> global_params_;
  TView<Real, 1, D> lj_lk_weights_;

  TView<Real, 1, D> output_;

  TView<int64_t, 1, tmol::Device::CPU> score_event_;
  TView<int64_t, 1, tmol::Device::CPU> annealer_event_;
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKRPERegistratorDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D> context_coords,
    TView<Int, 2, D> context_coord_offsets,
    TView<Int, 2, D> context_block_type,
    TView<Vec<Real, 3>, 1, D> alternate_coords,
    TView<Int, 1, D> alternate_coord_offsets,
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
    TView<Int, 1, D> block_type_n_atoms,
    TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
    TView<Int, 2, D> block_type_heavy_atoms_in_tile,

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
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,
    TView<Real, 1, D> lj_lk_weights,
    TView<Real, 1, D> output,

    TView<int64_t, 1, tmol::Device::CPU> score_event,
    TView<int64_t, 1, tmol::Device::CPU> annealer_event,

    TView<int64_t, 1, tmol::Device::CPU> annealer) -> void {
  using tmol::pack::sim_anneal::compiled::RPECalc;
  using tmol::pack::sim_anneal::compiled::SimAnnealer;

  int64_t annealer_uint = annealer[0];
  SimAnnealer* sim_annealer = reinterpret_cast<SimAnnealer*>(annealer_uint);
  std::shared_ptr<RPECalc> calc =
      std::make_shared<LJLKRPECPUCalc<DeviceDispatch, D, Real, Int>>(
          context_coords,
          context_coord_offsets,
          context_block_type,
          alternate_coords,
          alternate_coord_offsets,
          alternate_ids,
          context_system_ids,
          system_min_bond_separation,
          system_inter_block_bondsep,
          system_neighbor_list,
          block_type_n_atoms,
          block_type_n_heavy_atoms_in_tile,
          block_type_heavy_atoms_in_tile,
          block_type_atom_types,
          block_type_n_interblock_bonds,
          block_type_atoms_forming_chemical_bonds,
          block_type_path_distance,
          type_params,
          global_params,
          lj_lk_weights,
          output,
          score_event,
          annealer_event);

  sim_annealer->add_score_component(calc);
}

template struct LJLKRPEDispatch<ForallDispatch, tmol::Device::CPU, float, int>;
template struct LJLKRPEDispatch<ForallDispatch, tmol::Device::CPU, double, int>;
template struct LJLKRPERegistratorDispatch<
    ForallDispatch,
    tmol::Device::CPU,
    float,
    int>;
template struct LJLKRPERegistratorDispatch<
    ForallDispatch,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
