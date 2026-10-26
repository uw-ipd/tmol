"""Conversion between structure files, AtomArrays and PoseStacks.

Every file path goes through an AtomArray: a file is read into one and a pose
is built from it, and a pose is written by way of one.
"""

import io
import tempfile
from pathlib import Path
from typing import IO, Literal, Optional, Union

import biotite.structure as struc
import numpy
import torch
from atomworks.io.utils.io_utils import CIFWriteConfig, to_cif_string
from atomworks.io.utils.link_chemistry import DEFAULT_VALENCE
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFCategory, CIFFile

from tmol.io._assemble import _atom_array_from_small_molecule
from tmol.io._cif import _atom_array_from_pdbx
from tmol.io._pose_stack_from_biotite import (
    atom_array_from_pose_stack,
    pose_stack_from_atom_array,
)
from tmol.pose import PoseStack

StructureFormat = Literal["pdb", "cif", "mol2", "sdf"]
READ_FORMATS = ("pdb", "cif", "mol2", "sdf")
WRITE_FORMATS = ("pdb", "cif")

StructureSource = Union[str, Path, IO]

_FORMAT_FOR_EXTENSION = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "cif",
    ".mmcif": "cif",
    ".bcif": "cif",
    ".mol2": "mol2",
    ".sdf": "sdf",
    ".mol": "sdf",
}
_COMPRESSION_EXTENSIONS = (".gz", ".gzip", ".zst")
_EXTENSION_FOR_FORMAT = {"pdb": ".pdb", "cif": ".cif", "mol2": ".mol2", "sdf": ".sdf"}
_MISSING_CIF_VALUE = ("", ".", "?")
_PDB_RECORDS = ("ATOM  ", "HETATM", "HEADER", "MODEL ", "CRYST1", "REMARK", "COMPND")


def _is_text(source) -> bool:
    """True for a str holding file contents rather than a path."""
    return isinstance(source, str) and "\n" in source


def _format_from_extension(path) -> Optional[str]:
    """The format a path's extension names, ignoring a compression suffix."""
    path = Path(path)
    if path.suffix.lower() in _COMPRESSION_EXTENSIONS:
        path = path.with_suffix("")
    return _FORMAT_FOR_EXTENSION.get(path.suffix.lower())


def _format_from_contents(text: str) -> Optional[str]:
    """The format text is written in, from its leading records."""
    if "@<TRIPOS>" in text:
        return "mol2"
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if lines[0].startswith("data_") or "\n_atom_site." in text:
        return "cif"
    if (
        "M  END" in text
        or "$$$$" in text
        or any(line.rstrip().endswith(("V2000", "V3000")) for line in lines[:5])
    ):
        return "sdf"
    if any(line.startswith(_PDB_RECORDS) for line in lines):
        return "pdb"
    return None


def _read_contents(source) -> Union[str, bytes]:
    """Text or bytes of a str-contents, stream or path source."""
    if _is_text(source):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    contents = source.read()
    if hasattr(source, "seek"):
        source.seek(0)
    return contents


def _infer_format(source) -> str:
    """The format of a source, from its extension or else its contents."""
    if isinstance(source, (str, Path)) and not _is_text(source):
        fmt = _format_from_extension(source)
        if fmt is not None:
            return fmt
    contents = _read_contents(source)
    if isinstance(contents, bytes):
        try:
            contents = contents.decode()
        except UnicodeDecodeError:
            # binary CIF is the only binary format read
            return "cif"
    fmt = _format_from_contents(contents)
    if fmt is None:
        raise ValueError(
            "Cannot infer the structure format; pass format= as one of "
            f"{READ_FORMATS}"
        )
    return fmt


def _validate_format(fmt, allowed):
    if fmt not in allowed:
        raise ValueError(f"format must be one of {allowed}, not {fmt!r}")


def atom_array_from_file(
    source: StructureSource,
    format: Optional[StructureFormat] = None,
    *,
    model: int = 1,
    assembly_id: Optional[str] = None,
    res_name: Optional[str] = None,
) -> struc.AtomArray:
    """Read a PDB, CIF, MOL2 or SDF structure into an AtomArray.

    Preserve supplied names, coordinates, hydrogens and covalent connections.
    Declared unresolved heavy atoms have NaN coordinates. Chemistry the file
    leaves out is supplemented from the component dictionary; a MOL2/SDF
    molecule is registered as its own component.

    Args:
        source: A path, the file's text, or a stream. CIF paths may be gzip or
            Zstandard compressed, or binary CIF.
        format: One of "pdb", "cif", "mol2", "sdf". Inferred from the extension,
            or else the contents, when None.
        model: Model to read from a PDB/CIF.
        assembly_id: Biological assembly to build from a CIF.
        res_name: Component code for a MOL2/SDF molecule; the file's when None.
    """
    fmt = format or _infer_format(source)
    _validate_format(fmt, READ_FORMATS)
    if (
        isinstance(source, (str, Path))
        and not _is_text(source)
        and _format_from_extension(source) == fmt
    ):
        return _atom_array_from_path(Path(source), fmt, model, assembly_id, res_name)
    # the readers dispatch on extension, so contents without one go via a file
    contents = _read_contents(source)
    binary = isinstance(contents, bytes)
    if binary:
        try:
            contents = contents.decode()
            binary = False
        except UnicodeDecodeError:
            pass
    suffix = ".bcif" if binary else _EXTENSION_FOR_FORMAT[fmt]
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / f"input{suffix}"
        if binary:
            path.write_bytes(contents)
        else:
            path.write_text(contents)
        return _atom_array_from_path(path, fmt, model, assembly_id, res_name)


def _atom_array_from_path(path, fmt, model, assembly_id, res_name):
    if fmt in ("mol2", "sdf"):
        return _atom_array_from_small_molecule(path, res_name=res_name)
    return _atom_array_from_pdbx(path, model=model, assembly_id=assembly_id)


def _pdb_text(atom_array) -> str:
    pdb_file = PDBFile()
    pdb_file.set_structure(atom_array)
    output = io.StringIO()
    pdb_file.write(output)
    return output.getvalue()


def _cif_text(atom_array) -> str:
    # chem_comp from the structure itself: a pose's components need not match the CCD
    config = CIFWriteConfig(
        id="tmol",
        include_entity_categories=True,
        chem_comp_source="array",
        warn_on_ccd_without_registry=False,
    )
    # an input's entity ids belong to its label chains, not the author chains
    # written here, so the writer assigns entities itself
    if "label_entity_id" in atom_array.get_annotation_categories():
        atom_array = atom_array.copy()
        atom_array.del_annotation("label_entity_id")
    # the writer takes models as a list of AtomArrays
    models = (
        list(atom_array) if isinstance(atom_array, struc.AtomArrayStack) else atom_array
    )
    cif = CIFFile.read(io.StringIO(to_cif_string(models, config=config)))
    _number_label_seq_ids(cif.block)
    _declare_links_without_valence(cif.block, atom_array)
    _restore_coordinate_precision(cif.block, atom_array)
    out = io.StringIO()
    cif.write(out)
    return out.getvalue()


_ORDER_FOR_BOND_TYPE = {
    struc.BondType.SINGLE: "sing",
    struc.BondType.DOUBLE: "doub",
    struc.BondType.TRIPLE: "trip",
}


def _declare_links_without_valence(block, atom_array):
    """Declare inter-residue bonds at atoms the reader cannot check by valence.

    The reader re-infers an undeclared polymer bond only at an atom with free
    valence, a hydrogen or a declared leaving group. An element with no default
    valence (P) needs the leaving group, which a component written from the
    structure lacks, so nucleic acid links would be refused on reading.

    TEMP until atomworks validates these links without CCD leaving groups.
    """
    model = (
        atom_array[0] if isinstance(atom_array, struc.AtomArrayStack) else atom_array
    )
    if model.bonds is None:
        return
    bonds = model.bonds.as_array()
    residue = struc.get_all_residue_positions(model)
    no_valence = ~numpy.isin(model.element, list(DEFAULT_VALENCE))
    i, j = bonds[:, 0], bonds[:, 1]
    links = bonds[
        (residue[i] != residue[j])
        & (no_valence[i] | no_valence[j])
        & numpy.isin(bonds[:, 2], list(_ORDER_FOR_BOND_TYPE))
    ]
    if not len(links):
        return
    site = block["atom_site"]
    n = len(model)
    atom_fields = {
        "label_asym_id": site["label_asym_id"].as_array(str)[:n],
        "label_comp_id": site["label_comp_id"].as_array(str)[:n],
        "label_seq_id": site["label_seq_id"].as_array(str)[:n],
        "label_atom_id": site["label_atom_id"].as_array(str)[:n],
    }
    ins_code = site["pdbx_PDB_ins_code"].as_array(str)[:n]

    def partner(p, atom):
        fields = {f"ptnr{p}_{f}": v[atom] for f, v in atom_fields.items()}
        fields[f"pdbx_ptnr{p}_PDB_ins_code"] = ins_code[atom]
        return fields

    def endpoint(row, p):
        return tuple(
            row[f"ptnr{p}_{f}"]
            for f in ("label_asym_id", "label_seq_id", "label_atom_id")
        ) + (row[f"pdbx_ptnr{p}_PDB_ins_code"],)

    rows = []
    if "struct_conn" in block:
        existing = block["struct_conn"]
        columns = {c: existing[c].as_array(str) for c in existing}
        rows = [dict(zip(columns, values)) for values in zip(*columns.values())]
    declared = {frozenset((endpoint(r, "1"), endpoint(r, "2"))) for r in rows}
    for a, b, bond_type in links:
        row = {
            "id": str(len(rows) + 1),
            "conn_type_id": "covale",
            "pdbx_value_order": _ORDER_FOR_BOND_TYPE[bond_type],
            **partner("1", a),
            **partner("2", b),
        }
        key = frozenset((endpoint(row, "1"), endpoint(row, "2")))
        if key not in declared:
            declared.add(key)
            rows.append(row)
    columns = list(dict.fromkeys(c for r in rows for c in r))
    block["struct_conn"] = CIFCategory(
        {c: numpy.array([r.get(c, "?") for r in rows]) for c in columns}
    )


def _restore_coordinate_precision(block, atom_array):
    """Overwrite the writer's three-decimal coordinates with the array's own.

    TEMP until atomworks allows precision control.
    """
    site = block["atom_site"]
    coord = atom_array.coord.reshape(-1, 3)
    if len(site["Cartn_x"].as_array(str)) != len(coord):
        raise ValueError("CIF atom_site rows do not match the written atoms")
    for axis, name in enumerate(("Cartn_x", "Cartn_y", "Cartn_z")):
        site[name] = coord[:, axis]


def _number_label_seq_ids(block):
    """Number label_seq_id 1..N in each chain with insertion codes.

    The writer copies the author residue number into label_seq_id, so residues
    told apart only by insertion code (Kabat 99, 99A) would share one label
    residue, which readers key on. Author numbering is left as written.
    """
    site = block["atom_site"]
    asym = site["label_asym_id"].as_array(str)
    seq = site["label_seq_id"].as_array(str)
    ins = site["pdbx_PDB_ins_code"].as_array(str)
    chains = set(asym[~numpy.isin(ins, _MISSING_CIF_VALUE)])
    if not chains:
        return
    label = {}
    count = dict.fromkeys(chains, 0)
    for key in zip(asym, seq, ins):
        if key[0] in chains and key[1] not in _MISSING_CIF_VALUE and key not in label:
            count[key[0]] += 1
            label[key] = str(count[key[0]])
    site["label_seq_id"] = numpy.array(
        [label.get(k, k[1]) for k in zip(asym, seq, ins)]
    )
    if "struct_conn" not in block:
        return
    conn = block["struct_conn"]
    for p in ("1", "2"):
        n = len(conn[f"ptnr{p}_label_seq_id"].as_array(str))
        code = f"pdbx_ptnr{p}_PDB_ins_code"
        keys = zip(
            conn[f"ptnr{p}_label_asym_id"].as_array(str),
            conn[f"ptnr{p}_label_seq_id"].as_array(str),
            conn[code].as_array(str) if code in conn else [""] * n,
        )
        conn[f"ptnr{p}_label_seq_id"] = numpy.array([label.get(k, k[1]) for k in keys])


_WRITERS = {"pdb": _pdb_text, "cif": _cif_text}


def file_from_atom_array(
    atom_array: Union[struc.AtomArray, struc.AtomArrayStack],
    path: Optional[Union[str, Path]] = None,
    format: Optional[Literal["pdb", "cif"]] = None,
) -> Union[str, Path]:
    """Write an AtomArray as PDB or CIF; return the text when path is None.

    A CIF carries the structure's bonds, so a component that is not a dictionary
    entry (a MOL2 ligand, a generated residue type) reads back with its chemistry.

    Args:
        atom_array: Structure to write.
        path: Where to write.
        format: "pdb" or "cif". Inferred from path's extension when None.
    """
    if format is None:
        if path is None:
            raise ValueError("Pass format= when writing without a path")
        format = _format_from_extension(path)
    _validate_format(format, WRITE_FORMATS)
    text = _WRITERS[format](atom_array)
    if path is None:
        return text
    path = Path(path)
    path.write_text(text)
    return path


def pose_stack_from_file(
    source: StructureSource,
    device: torch.device,
    format: Optional[StructureFormat] = None,
    *,
    model: int = 1,
    assembly_id: Optional[str] = None,
    residue_start: Optional[int] = None,
    residue_end: Optional[int] = None,
    **kwargs,
):
    """Build a PoseStack from a PDB, CIF, MOL2 or SDF structure.

    Reads with atom_array_from_file and builds with pose_stack_from_atom_array,
    which takes the remaining keyword arguments. residue_start and residue_end
    select a half-open range of residues.
    """
    array = atom_array_from_file(source, format, model=model, assembly_id=assembly_id)
    if residue_start is not None or residue_end is not None:
        boundaries = struc.get_residue_starts(array, add_exclusive_stop=True)
        start, stop, _ = slice(residue_start, residue_end).indices(len(boundaries) - 1)
        array = array[boundaries[start] : boundaries[stop]]
    return pose_stack_from_atom_array(array, device, **kwargs)


def file_from_pose_stack(
    pose_stack: PoseStack,
    path: Optional[Union[str, Path]] = None,
    format: Optional[Literal["pdb", "cif"]] = None,
    **kwargs,
) -> Union[str, Path]:
    """Write a PoseStack as PDB or CIF; return the text when path is None.

    Converts with atom_array_from_pose_stack, which takes the remaining keyword
    arguments, and writes with file_from_atom_array.
    """
    return file_from_atom_array(
        atom_array_from_pose_stack(pose_stack, **kwargs), path, format
    )
