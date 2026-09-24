import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

NCAA_FIXTURE_CLASS = {
    "6dmz_mod_d.cif": "mirror_image_control",
    "6dmz_mod_l.cif": "mirror_image_control",
    "beta_peptide_3c3g.cif": "nonstandard_backbone",
    "capped_peptide_ace_nh2.cif": "terminal_caps",
    "capped_peptide_ace_nme.cif": "terminal_caps",
    "chromophore_nrq_3svu.cif": "covalent_chromophore",
    "collagen_hyp_1bkv.cif": "alpha_amino_acid_modification",
    "gamma_peptide_1gac.cif": "nonstandard_backbone",
    "na_dna_5mc_1d17.cif": "modified_nucleic_acid",
    "na_dna_8og_183d.cif": "modified_nucleic_acid",
    "na_dna_ttd_1ttd.cif": "modified_nucleic_acid",
    "na_rna_2ome_310d.cif": "modified_nucleic_acid",
    "na_rna_psu_1bzt.cif": "modified_nucleic_acid",
    "nmethyl_peptide_6mvz.cif": "nonstandard_backbone",
    "phosphopeptide_5ema.cif": "alpha_amino_acid_modification",
}

DATA = Path(__file__).parent / "data"


def test_atomworks_regression_manifest_matches_fixture_bytes() -> None:
    """Require one valid provenance row for every bundled structure."""
    fixture_dir = DATA / "atomworks_regressions"
    provenance = json.loads((fixture_dir / "provenance.json").read_text())
    manifest_names = [row["fixture"] for row in provenance]
    directory_names = {
        path.name
        for pattern in ("*.cif", "*.cif.gz")
        for path in fixture_dir.glob(pattern)
    }

    assert len(manifest_names) == len(set(manifest_names))
    assert set(manifest_names) == directory_names

    for row in provenance:
        contents = (fixture_dir / row["fixture"]).read_bytes()
        if "compressed_sha256" in row:
            assert hashlib.sha256(contents).hexdigest() == row["compressed_sha256"]
            contents = gzip.decompress(contents)
        assert hashlib.sha256(contents).hexdigest() == row["sha256"]
        assert len(contents) == row["bytes"]


def test_ncaa_fixture_inventory_has_documented_classification() -> None:
    """Keep all 15 README fixture roles explicit without claiming parity."""
    fixture_dir = DATA / "ncaa_fixtures"
    assert {path.name for path in fixture_dir.glob("*.cif")} == set(NCAA_FIXTURE_CLASS)
    assert Counter(NCAA_FIXTURE_CLASS.values()) == {
        "alpha_amino_acid_modification": 2,
        "covalent_chromophore": 1,
        "mirror_image_control": 2,
        "modified_nucleic_acid": 5,
        "nonstandard_backbone": 3,
        "terminal_caps": 2,
    }
