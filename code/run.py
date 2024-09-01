import os
import sys
import warnings
import torch
import scanpy as sc
from constants import *
from preprocessing import preprocess
from predictions import train


def setup_libraries() -> None:
    """
    Set up the libraries to be used in the project.

    :return: None
    """
    warnings.filterwarnings('ignore')

    os.environ['TORCH'] = torch.__version__
    print(f"Currently running with torch version '{torch.__version__}'. "
          f"This model was last tested successfully with version '{LAST_TESTED_TORCH_VERSION}'.\n")

    # Use all available threads and cores
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

    # Set scanpy to use all available cores
    sc.settings.n_jobs = -1

    return


def arguments() -> (str, str, str, str):
    """
    Extract and validate the command line arguments to decide how to run the model.

    :return: The input matrix name, folder name, input matrix path, and preprocessed folder path.
    """
    # Extract command line arguments
    input_matrix_name = sys.argv[1]
    folder_name = sys.argv[2]

    # Build the paths to the input matrix and preprocessed folder
    input_matrix_path = f"{RAW_DATA_PATH}/{folder_name}/{input_matrix_name}"
    preprocessed_folder_path = f"{PREPROCESSED_DATA_PATH}/{folder_name}"

    return input_matrix_name, folder_name, input_matrix_path, preprocessed_folder_path


def print_row(file_name: str, file_type: str, status: str, expected_path: str, notes: str) -> None:
    """
    Print a row of the file path validation table.

    :param file_name: The name of the file.
    :param file_type: The type of the file.
    :param status: The status of the file.
    :param expected_path: The relative path to the file.
    :param notes: Any additional notes to include.

    :return: None
    """
    print(f"{file_name:<30} {file_type:<15} {status:<15} {expected_path:<70} {notes}")

    return


def validate_single_file_path(file_type: str, file_name: str, expected_path: str, notes: str, create_path: bool,
                              input_path_error: bool) -> int:
    """
    Validate that the file path provided is valid.

    :param file_type: The type of file to validate.
    :param file_name: The name of the file to validate.
    :param expected_path: The relative path to the file.
    :param notes: Any additional notes to include if the file is not found.
    :param create_path: Whether to create the path if it does not exist.
    :param input_path_error: Whether there were errors with the input paths.

    :return: None
    """

    # Reformated expected path to be more readable
    readable_expected_path = expected_path.replace("..", "<PROJECT_ROOT>")

    # If file exists at the relative path, print a found message and don't increment the failed paths count
    if os.path.exists(f"{expected_path}/{file_name}"):
        print_row(file_name, file_type, "FOUND", f"{readable_expected_path}/{file_name}",
                  "No action required.")
        return 0

    # Otherwise, print an error message and increment the failed paths count
    else:
        # If the input path is invalid, see if the path can be created automatically
        if create_path:

            # Check to see that there were no errors with the input paths
            if input_path_error:
                print_row(file_name, file_type, "SKIPPED", f"{readable_expected_path}/{file_name}",
                          "Please fix errors with input file and folder paths first.")
                return 1

            # Try to create the directory automatically
            if os.system("mkdir -p " + f"{expected_path}/{file_name}") == -1:
                print_row(file_name, file_type, "FATAL ERROR", f"{readable_expected_path}/{file_name}",
                          "The directory could not be automatically created. Please check path the project structure.")
                return 1

        # Base case: print an error message and increment the failed paths count
        print_row(file_name, file_type, "ERROR", f"{readable_expected_path}/{file_name}", notes)

        return 1


def validate_all_file_paths(input_matrix_name: str, folder_name: str) -> None:
    """
    Validate that all the file paths provided are valid, as well as output folders.

    :param input_matrix_name: The name of the input matrix file.
    :param folder_name: The name of the folder containing the input matrix file.

    :return: None
    """
    # Count the number of paths that were failed to be validated
    failed_paths_count = 0

    # Assume that the input paths are valid until proven otherwise
    input_path_error = False

    print("Argument 'check' provided. Validating paths then exiting...\n")

    # Print the table headers
    print_row("Name", "FileType", "Status", "ExpectedPath", "Notes")
    print_row("-----------", "-----------", "---------", "---------------",
              "--------")

    # Validate the folder name
    failed_paths_count += validate_single_file_path("folder", folder_name, RAW_DATA_PATH,
                                                    "Check the spelling of your arguments and/or that files have "
                                                    "been placed in the correct directory.",
                                                    False, input_path_error)

    # Validate the input matrix file path
    failed_paths_count += validate_single_file_path("file", input_matrix_name, f"{RAW_DATA_PATH}/{folder_name}",
                                                    "Check the spelling of your arguments and/or that files have "
                                                    "been placed in the correct directory.",
                                                    False, input_path_error)

    # If there were errors with the input paths, don't automatically create the preprocessed and output folders
    if failed_paths_count > 0:
        input_path_error = True

    # Validate the preprocessed folder path
    failed_paths_count += validate_single_file_path("folder", folder_name, PREPROCESSED_DATA_PATH,
                                                    "No preprocessed folder found so one was automatically "
                                                    "created. Please run validate paths again to confirm.",
                                                    True, input_path_error)

    # Validate the output folder path
    failed_paths_count += validate_single_file_path("folder", folder_name, OUTPUT_DATA_PATH,
                                                    "No output folder found so one was automatically "
                                                    "created. Please run validate paths again to confirm.",
                                                    True, input_path_error)

    # Validate the lr_nodes file path
    failed_paths_count += validate_single_file_path("file", LR_NODES_FILE, LR_NODES_PATH,
                                                    "Check that the location of the LR nodes file is correct and "
                                                    "hasn't been moved.",
                                                    False, input_path_error)

    # Validate the omnipath network intercell interactions file
    failed_paths_count += validate_single_file_path("file", INTERCELL_INTERACTIONS_FILE, OMNIPATH_DATABASE_PATH,
                                                    "Check that the location of the this Omnipath database file "
                                                    "is correct and hasn't been moved.",
                                                    False, input_path_error)

    # Validate the omnipath network complexes interactions file
    failed_paths_count += validate_single_file_path("file", COMPLEXES_FILE, OMNIPATH_DATABASE_PATH,
                                                    "Check that the location of the this Omnipath database file "
                                                    "is correct and hasn't been moved.",
                                                    False, input_path_error)

    # Validate the omnipath network kegg pathways interactions file
    failed_paths_count += validate_single_file_path("file", KEGG_PATHWAYS_FILE, OMNIPATH_DATABASE_PATH,
                                                    "Check that the location of the this Omnipath database file "
                                                    "is correct and hasn't been moved.",
                                                    False, input_path_error)

    # Validate the omnipath network consensus interactions file
    failed_paths_count += validate_single_file_path("file", CONSENSUS_OMNIPATH_FILE, OMNIPATH_DATABASE_PATH,
                                                    "Check that the location of the this Omnipath database file "
                                                    "is correct and hasn't been moved.",
                                                    False, input_path_error)

    # Validate the omnipath network database file
    failed_paths_count += validate_single_file_path("file", OMNIPATH_DATABASE_FILE, OMNIPATH_DATABASE_PATH,
                                                    "Check that the location of the this Omnipath database file "
                                                    "is correct and hasn't been moved.",
                                                    False, input_path_error)

    # If there were failed paths, print a conclusion message and exit the program
    if failed_paths_count > 0:
        print(f"\n{failed_paths_count} paths failed validation. Please check the notes above for more information and "
              f"try again. Stopping program execution.\n")
        sys.exit(1)

    # If all paths were validated successfully, print a success message and continue with the program
    print("\nAll paths validated successfully. Continuing with program execution...\n")

    return


def main() -> None:
    """
    The main driver function for GraphComm. Handles setting up the libraries, handling command line arguments, and
    calling the preprocessing and training scripts as requested.

    :return: None
    """
    # Set up the libraries
    setup_libraries()

    # Handle command line arguments
    input_matrix_name, folder_name, input_matrix_path, preprocessed_folder_path = arguments()

    # Validate the file paths
    validate_all_file_paths(input_matrix_name, folder_name)

    # Get the path of the current file relative to the root directory
    current_file_path = os.path.realpath(__file__)
    print(current_file_path)

    # Call the preprocessing script
    print("Preprocessing the input matrix...")
    preprocess(input_matrix_path, preprocessed_folder_path,
               f"{LR_NODES_PATH}/{LR_NODES_FILE}",
               f"{OMNIPATH_DATABASE_PATH}/{INTERCELL_INTERACTIONS_FILE}")
    print("Preprocessing complete.\n")

    # Now call the training script
    print("Training the model...")
    train(preprocessed_folder_path,
          f"{OUTPUT_DATA_PATH}/{folder_name}",
          f"{OMNIPATH_DATABASE_PATH}/{COMPLEXES_FILE}",
          f"{OMNIPATH_DATABASE_PATH}/{KEGG_PATHWAYS_FILE}",
          f"{OMNIPATH_DATABASE_PATH}/{CONSENSUS_OMNIPATH_FILE}",
          f"{OMNIPATH_DATABASE_PATH}/{OMNIPATH_DATABASE_FILE}")
    print("Training complete.")

    return


if __name__ == "__main__":
    main()
