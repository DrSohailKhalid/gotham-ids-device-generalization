# Original dataset

The raw Gotham Dataset 2025 is not redistributed in this repository.

Official dataset record: https://doi.org/10.5281/zenodo.14502760

The published dataset comprises 78 device-level traffic captures. The analysis
pipeline queries the Zenodo record, obtains the available CSV/archived files and
verifies published file checksums when available. It does not require raw PCAPs.

The reported experiment uses up to 8,000 retained records per device after
causal feature derivation, yielding a working benchmark of 404,324 packets.
This benchmark is a nonuniform, deliberately sampled subset, not the full
distribution of the public dataset.

The data are provided by their original authors. Consult the Zenodo record for
its current license and redistribution terms. Do not commit the downloaded
archive, extracted device files or intermediate packet-level tables.
