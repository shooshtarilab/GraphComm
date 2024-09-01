# This file contains all the constants used in the project and is imported where necessary.
# They are all relative to run.py, as the current working directory is set to where run.py is located, and all paths
# are relative to that location (even if run.py calls a function in another file that sits in a different directory).

RAW_DATA_PATH = "../data/Raw_Data"
PREPROCESSED_DATA_PATH = "../data/Preprocessed_Data"
OUTPUT_DATA_PATH = "../data/Output_Data"

LR_NODES_PATH = "../data/LR_database"
LR_NODES_FILE = "intercell_nodes.csv"

OMNIPATH_DATABASE_PATH = "../data/LR_database"
INTERCELL_INTERACTIONS_FILE = "intercell_interactions.csv"
COMPLEXES_FILE = "complexes.csv"
KEGG_PATHWAYS_FILE = "kegg_pathways.csv"
CONSENSUS_OMNIPATH_FILE = "consensus_Omnipath.csv"
OMNIPATH_DATABASE_FILE = "Omnipath_database.csv"

LAST_TESTED_TORCH_VERSION = "2.2.1+cu121"
