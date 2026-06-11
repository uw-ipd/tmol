mol2fn=./designs.prot.mmff94.mol2
python /home/gzhou/git/Rosetta/rosetta_rosettavs/source/scripts/python/public/generic_potential/mol2genparams.py -s $mol2fn --outdir params --resname LG1 --typenm automol2name --rename_atoms --multimol2 --infer_atomtypes --no_pdb
