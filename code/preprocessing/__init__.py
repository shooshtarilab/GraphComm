import pandas as pd
import anndata as ad
import scanpy as sc
import datatable as dt


def preprocess(input_matrix_path: str, preprocessed_folder_path: str, lr_nodes_path: str,
               intercell_interactions_path: str) -> None:
    """
    Main preprocessing function that handles the loading, preprocessing, and saving of gene expression data,
    metadata, and ligand-receptor interactions for later training.

    :param input_matrix_path: Path to the CSV file containing the gene expression matrix.
    :param preprocessed_folder_path: Path to the directory where the preprocessed files will be saved.
    :param lr_nodes_path: Path to the CSV file containing ligands and receptors.
    :param intercell_interactions_path: Path to the CSV file containing known intercellular interactions.

    :return: None
    """
    print("Loading and preprocessing the input matrix...")

    # Load and preprocess the input matrix
    adata = load_and_preprocess(input_matrix_path)

    print("Done\nCreating AnnData metadata for the cells...")

    # Create metadata for the cells
    adata, meta, cell_type_df, matrix = create_metadata(adata)

    print("Done\nLoading in ligand-receptor and omnipath databases...")

    # Load the ligand-receptor and omnipath databases
    lr_nodes, omnipath_network, ligands, receptors = load_lr_database(lr_nodes_path, intercell_interactions_path)

    print("Done\nComparing expressed genes with the ligand-receptor database...")

    # Compare expressed genes with the ligand-receptor database
    ligand_list, receptor_list, new_cell_df = compare_genes_with_lr_database(cell_type_df, ligands, receptors)

    print("Done\nCreating metadata DataFrame with cells as indices...")

    # Create a DataFrame of nodes
    ligand_list, receptor_list, lr_nodes, nodes = create_nodes_df(ligand_list, receptor_list, cell_type_df, lr_nodes)

    print("Done\nAdding interactions between nodes to the DataFrame...")

    # Create a DataFrame of interactions
    meta, nodes, lr_nodes, interactions = create_interactions_df(meta, new_cell_df, nodes, lr_nodes, omnipath_network)

    print("Done\nFiltering data to include only relevant ligands and receptors...")

    # Filter the data to include only relevant ligands and receptors
    lr_nodes, omnipath_network, interactions = filter_data(lr_nodes, omnipath_network, ligand_list, receptor_list,
                                                           interactions)
    print("Done\nSaving the preprocessed files...")

    # Save the preprocessed files
    save_preprocessed_files(preprocessed_folder_path, nodes, interactions, meta, matrix, lr_nodes, omnipath_network)

    print("Done")


def load_and_preprocess(input_matrix_path: str) -> ad.AnnData:
    """
    Load a gene expression matrix from a CSV file and preprocess it for further analysis. Preprocessing steps
    include: converting to pandas DataFrame, transposing the matrix, filtering genes, normalizing total expression,
    applying log transformation, computing neighbors, and clustering cells using the Leiden algorithm.

    :param input_matrix_path: Path to the CSV file containing the gene expression matrix.

    :return: An AnnData object containing the preprocessed gene expression data.
    """
    # Load the CSV data using datatable for faster reading
    dt_frame = dt.fread(input_matrix_path)
    matrix = dt_frame[:, 1:].to_pandas()

    # Get the gene names from the first column
    gene_names = dt_frame[:, 0].to_list()[0]

    # Create the AnnData object
    adata = ad.AnnData(X=matrix.T)

    # Assign gene names to .var_names if rows represent genes
    gene_names_str = [str(name) for name in gene_names]
    adata.var_names = gene_names_str

    # Preprocessing with Scanpy functions
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)

    return adata


def create_metadata(adata: ad.AnnData) -> (ad.AnnData, pd.DataFrame, dict, pd.DataFrame):
    """
    Create a DataFrame containing metadata for cells based on their clustering (Leiden algorithm). Updates the
    AnnData object with this metadata and creates a list of matrices for each cell group based on their mean expression.

    :param adata: An AnnData object with gene expression data and clustering information.

    :return: An AnnData object with updated metadata, a DataFrame containing cell metadata, a dictionary of matrices
    for each cell group, and an adjusted gene expression matrix.
    """
    # Create a DataFrame containing cell metadata (Leiden cluster labels)
    meta = pd.DataFrame({"cell": adata.obs["leiden"].index.tolist(), "labels": adata.obs["leiden"].tolist()})
    meta.index = meta["cell"].tolist()

    # Update the `adata.obs` attribute with the cell metadata
    adata.obs = meta

    # Transpose and adjust the matrix DataFrame
    matrix = pd.DataFrame(adata.X.transpose(), columns=adata.obs.index.tolist(), index=adata.var.index.tolist())

    cell_groups = meta['labels'].unique().tolist()  # Get unique cluster labels
    matrix_list = {}
    for i in cell_groups:
        cells = meta[meta["labels"] == i].index.tolist()  # Get cells in each cluster
        temp_matrix = matrix[cells]  # Subset matrix for each cluster
        matrix_list[i] = (temp_matrix.mean(axis=1)[temp_matrix.mean(axis=1) >= 0].index.tolist())

    return adata, meta, matrix_list, matrix


def load_lr_database(lr_nodes_path: str,
                     intercell_interactions_path: str) -> (pd.DataFrame, pd.DataFrame, list, list):
    """
    Load the ligand-receptor database from CSV files and extract ligands and receptors for further preprocessing.

    :param lr_nodes_path: Path to the CSV file containing ligands and receptors.
    :param omnipath_database_path: Path to the CSV file containing known interactions between ligands and receptors.

    :return: A DataFrame containing identifiers for ligands and receptors, a DataFrame containing known interactions
    between ligands and receptors, a lists of ligands, and a list of receptors.
    """

    # LR_nodes represent ligands and receptors
    lr_nodes = pd.read_csv(lr_nodes_path, index_col=0)

    # Omnipath_network represents known interactions between them
    omnipath_network = pd.read_csv(intercell_interactions_path, index_col=0)

    ligands = lr_nodes[lr_nodes["category"] == "Ligand"]["identifier"].tolist()
    receptors = lr_nodes[lr_nodes["category"] == "Receptor"]["identifier"].tolist()

    return lr_nodes, omnipath_network, ligands, receptors


def compare_genes_with_lr_database(cell_type_df: dict, ligands: list, receptors: list) -> (list, list, dict):
    """
    Compare expressed genes in each cell type against the ligand-receptor database to identify which ligands and
    receptors are expressed in the input gene expression data. Create a new DataFrame with this information.

    :param cell_type_df: A dictionary containing cell type labels as keys and expressed genes as values.
    :param ligands: A list of ligands from the ligand-receptor database.
    :param receptors: A list of receptors from the ligand-receptor database.

    :return: A list of expressed ligands, a list of expressed receptors, and a new DataFrame containing expressed
    ligands and receptors mapped to each cell type.
    """
    ligand_list = []
    receptor_list = []
    new_cell_df = {}

    for i in cell_type_df.keys():
        # Find ligands and receptors expressed within each cell type
        ligand_list.extend(list(set(ligands) & set(cell_type_df[i])))
        receptor_list.extend(list(set(receptors) & set(cell_type_df[i])))

        # Store expressed ligands and receptors for each cell type (cluster)
        new_cell_df[i] = [
            list(set(ligands) & set(cell_type_df[i])),  # Expressed ligands
            list(set(receptors) & set(cell_type_df[i])),  # Expressed receptors
        ]

    for i in new_cell_df.keys():
        # For each key, get the first element of the value list (assumed to be ligands)
        # and append "_Ligand" to each ligand to create a unique identifier
        new_cell_df[i][0] = [j + "_Ligand" for j in new_cell_df[i][0]]

        # Get the second element of the value list (assumed to be receptors)
        # and append "_Receptor" to each receptor to create a unique identifier
        new_cell_df[i][1] = [j + "_Receptor" for j in new_cell_df[i][1]]

    return ligand_list, receptor_list, new_cell_df


def create_nodes_df(ligand_list: list, receptor_list: list, cell_type_df: dict,
                    lr_nodes: pd.DataFrame) -> (list, list, pd.DataFrame, pd.DataFrame):
    """
    Create a DataFrame of nodes that includes cell groups, ligands, and receptors, each with a unique identifier.

    :param ligand_list: A list of expressed ligands.
    :param receptor_list: A list of expressed receptors.
    :param cell_type_df: A dictionary containing cell type labels as keys and expressed genes as values.
    :param lr_nodes: A DataFrame containing ligands and receptors from the ligand-receptor database.

    :return: An updated list of ligands, an updated list of receptors, an updated DataFrame of ligands and receptor
    nodes with unique identifiers, and a DataFrame of nodes.
    """
    ligand_list = list(set(ligand_list))  # Get unique ligands
    receptor_list = list(set(receptor_list))  # Get unique receptors

    # Rename ligands/receptors to include "Ligand" / "Receptor" suffixes
    ligand_list = [i + "_Ligand" for i in ligand_list]
    receptor_list = [i + "_Receptor" for i in receptor_list]

    # Create the nodes DataFrame
    nodes = pd.concat([
        pd.DataFrame(
            {"category": ["Cell Group"] * len(list(cell_type_df.keys())), "identifier": list(cell_type_df.keys())}),
        pd.DataFrame({"category": ["Ligand"] * len(ligand_list), "identifier": ligand_list}),
        pd.DataFrame({"category": ["Receptor"] * len(receptor_list), "identifier": receptor_list})
    ])

    # Create unique IDs for each node
    new_identifier = [row["identifier"] + "_" + row["category"] for index, row in lr_nodes.iterrows()]
    lr_nodes["identifier"] = new_identifier

    # Add ID column to the nodes DataFrame
    nodes["Id"] = range(0, nodes.shape[0])
    nodes = nodes[["Id", "category", "identifier"]]
    nodes.index = nodes.index.astype('int')
    nodes["Id"] = nodes["Id"].astype('int')

    return ligand_list, receptor_list, lr_nodes, nodes


def create_interactions_df(meta: pd.DataFrame, new_cell_df: dict, nodes: pd.DataFrame, lr_nodes: pd.DataFrame,
                           omnipath_network: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Create a DataFrame detailing the interactions between nodes, including cell groups and their expressed
    ligands/receptors, as well as known interactions from the omnipath_network.

    :param meta: A DataFrame containing metadata for cells.
    :param new_cell_df: A dictionary containing cell type labels as keys and expressed ligands and receptors as values.
    :param nodes: A DataFrame containing node information.
    :param lr_nodes: A DataFrame containing ligands and receptors with unique identifiers.
    :param omnipath_network: A DataFrame containing known interactions between ligands and receptors.

    :return: An updated metadata DataFrame with cells as the index, an updated nodes DataFrame with unique numerical
    IDs, an updated ligand-receptor nodes DataFrame, and a DataFrame detailing interactions between nodes.
    """
    # Reset index of meta for easy access
    meta.index = meta["cell"].tolist()

    # Create the interactions DataFrame
    interactions = pd.DataFrame({"Src": [], "Dst": [], "Weight": [], "edge_type": []})

    # Aligning DataFrames with Node IDs
    lr_nodes.index = lr_nodes["Id"].tolist()

    # Increment 'Src' and 'Dst'
    omnipath_network["Src"] += 1
    omnipath_network["Dst"] += 1
    omnipath_network["Src"] = lr_nodes.loc[omnipath_network["Src"].tolist()]["identifier"].tolist()
    omnipath_network["Dst"] = lr_nodes.loc[omnipath_network["Dst"].tolist()]["identifier"].tolist()

    # Add interactions between cell groups and their expressed ligands/receptors
    source_list = []
    dest_list = []
    weight_list = []
    edge_type_list = []

    for i in new_cell_df.keys():
        source_list.extend([i] * (len(new_cell_df[i][0]) + len(new_cell_df[i][1])))  # Cell group as source
        dest_list.extend(new_cell_df[i][0])  # Ligands as destinations
        dest_list.extend(new_cell_df[i][1])  # Receptors as destinations
        weight_list.extend([1] * (len(new_cell_df[i][0]) + len(new_cell_df[i][1])))  # Sample weight
        edge_type_list.extend([1] * (len(new_cell_df[i][0]) + len(new_cell_df[i][1])))  # Sample edge type

    interactions["Src"] = source_list
    interactions["Dst"] = dest_list
    interactions["Weight"] = weight_list
    interactions["edge_type"] = edge_type_list

    # Map node identifiers to IDs for consistency
    nodes.index = nodes["identifier"].tolist()

    nodes = nodes.drop_duplicates("identifier")
    nodes["Id"] = range(0, nodes.shape[0])

    interactions["Src"] = nodes.loc[interactions["Src"].tolist()]["Id"].tolist()
    interactions["Dst"] = nodes.loc[interactions["Dst"].tolist()]["Id"].tolist()

    return meta, nodes, lr_nodes, interactions


def filter_data(lr_nodes: pd.DataFrame, omnipath_network: pd.DataFrame, ligand_list: list, receptor_list: list,
                interactions: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Filter ligands, receptors, and interactions to only include relevant items based on the dataset analysis. Updates
    identifiers and aligns dataframes to include only these filtered items.

    :param lr_nodes: A DataFrame containing ligands and receptors with unique identifiers.
    :param omnipath_network: A DataFrame containing known interactions between ligands and receptors.
    :param ligand_list: A list of expressed ligands to be filtered.
    :param receptor_list: A list of expressed receptors to be filtered.
    :param interactions: A DataFrame of interactions between nodes to be updated.

    :return: A filtered DataFrame of ligand and receptor nodes, a filtered DataFrame of ligand-receptor interactions,
    and an updated DataFrame of interactions with filtered nodes.
    """
    # Setting the 'identifier' column as the index for easier referencing
    lr_nodes.index = lr_nodes["identifier"].tolist()

    # Keep only ligands and receptors present in the analyzed dataset
    lr_nodes = lr_nodes[(lr_nodes["identifier"].isin(ligand_list)) | (lr_nodes["identifier"].isin(receptor_list))]

    # Filter interactions to include only those with sources and destinations present in the filtered lr_nodes
    omnipath_network = omnipath_network[(omnipath_network["Src"].isin(lr_nodes["identifier"].tolist())) & (
        omnipath_network["Dst"].isin(lr_nodes["identifier"].tolist()))]

    # Update the lists of ligand and receptor identifiers
    ligand_list = omnipath_network["Src"].tolist()
    receptor_list = omnipath_network["Dst"].tolist()

    # Further refine LR_nodes to include only the remaining ligands and receptors
    lr_nodes = lr_nodes[(lr_nodes["identifier"].isin(ligand_list)) | (lr_nodes["identifier"].isin(receptor_list))]

    # Filter Omnipath_network to include interactions consistent with the refined LR_nodes
    omnipath_network = omnipath_network[(omnipath_network["Src"].isin(lr_nodes["identifier"].tolist())) & (
        omnipath_network["Dst"].isin(lr_nodes["identifier"].tolist()))]

    # Assign unique numerical IDs to each node in LR_nodes
    lr_nodes["Id"] = range(0, lr_nodes.shape[0])

    # Set the 'identifier' column back as the index
    lr_nodes.index = lr_nodes["identifier"].tolist()

    # Replace source node identifiers in Omnipath_network with their corresponding IDs from LR_nodes
    omnipath_network["Src"] = lr_nodes.loc[omnipath_network["Src"].tolist()]["Id"].tolist()

    # Replace destination node identifiers in Omnipath_network with their corresponding IDs from LR_nodes
    omnipath_network["Dst"] = lr_nodes.loc[omnipath_network["Dst"].tolist()]["Id"].tolist()

    # Assign a uniform edge type to all interactions (replace 1 with a meaningful value if necessary)
    interactions["edge_type"] = 1

    return lr_nodes, omnipath_network, interactions


def save_preprocessed_files(preprocessed_folder_path: str, nodes: pd.DataFrame, interactions: pd.DataFrame,
                            meta: pd.DataFrame, matrix: pd.DataFrame, lr_nodes: pd.DataFrame,
                            omnipath_network: pd.DataFrame) -> None:
    """
    Save preprocessed files to a specified directory. Creates the directory if it does not exist and saves various
    DataFrames as CSV files.

    :param preprocessed_folder_path: Path to the directory where the preprocessed files will be saved.
    :param nodes: A DataFrame containing node information.
    :param interactions: A DataFrame containing interactions between nodes.
    :param meta: A DataFrame containing metadata for cells.
    :param matrix: A DataFrame containing gene expression data.
    :param lr_nodes: A DataFrame containing ligands and receptors with unique identifiers.
    :param omnipath_network: A DataFrame containing known interactions between ligands and receptors.

    :return: None
    """
    # Save the data to CSV files in the output directory
    nodes.to_csv(preprocessed_folder_path + "/nodes.csv")
    interactions.to_csv(preprocessed_folder_path + "/interactions.csv")
    meta.to_csv(preprocessed_folder_path + "/meta.csv")
    matrix.to_csv(preprocessed_folder_path + "/matrix.csv")
    lr_nodes.to_csv(preprocessed_folder_path + "/lr_nodes.csv")
    omnipath_network.to_csv(preprocessed_folder_path + "/omnipath_network.csv")
