# branchRatio

This repository contains code supporting the results presented in the paper "Measuring competing outcomes of a single-molecule reaction reveals classical Arrhenius chemical kinetics".

`data` contains the raw data of single toluene molecule manipulation experiments carried out on the Si(111)-7x7 surface in csv format.
The experiments span the injection bias voltage range of +1.4 eV to +2.2 eV, the injection current range of 25 pA to 900 pA, and three separate injection locations: on top of an adatoms, a molecule, and a restaom.

The analysis code extracts manipulation probabilities and branching ratios as well as fitting to spectroscopy data above faulted corner and toluene faulted middle sites. Full description of the code capability can be found in the comment block at the top of script.

## Usage

`python3 tolueneBRatioAnalysis.py`

Figures will be produced and saved to the `figures` directory.

Some extracted parameters will be printed to `stdout` (terminal).

When looking to extract rates with the 'double fit method' (see Suplementary Note 2) stayed molecules must be excluded.
