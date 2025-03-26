import os
import pandas as pd
import gzip
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_pkl_to_csv(input_dir: str, output_dir: str = None) -> None:
    """
    Convert all pickle files in the input directory to CSV format.
    
    Args:
        input_dir: Directory containing pickle files (.pkl or .pkl.gz)
        output_dir: Directory to save CSV files (defaults to input_dir if None)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all pickle files
    pkl_files = list(input_path.glob("*.pkl")) + list(input_path.glob("*.pkl.gz"))
    
    if not pkl_files:
        logger.warning(f"No pickle files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pkl_files)} pickle files to convert")
    
    for pkl_file in pkl_files:
        try:
            # Load the pickle file
            if pkl_file.suffix == '.gz':
                logger.info(f"Loading compressed pickle file: {pkl_file}")
                with gzip.open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                logger.info(f"Loading pickle file: {pkl_file}")
                data = pd.read_pickle(pkl_file)
            
            # Convert to DataFrame if not already
            if not isinstance(data, pd.DataFrame):
                logger.warning(f"File {pkl_file} does not contain a pandas DataFrame. Attempting to convert.")
                data = pd.DataFrame(data)
            
            # Create output filename
            csv_filename = pkl_file.stem + '.csv'
            csv_path = output_path / csv_filename
            
            # Save as CSV with index as a column
            logger.info(f"Saving to CSV: {csv_path}")
            # Reset index to make it a column in the CSV
            data = data.reset_index()
            data.to_csv(csv_path, index=False)
            logger.info(f"Successfully converted {pkl_file} to {csv_path} with shape {data.shape}")
        except Exception as e:
            logger.error(f"Error converting {pkl_file}: {str(e)}")

if __name__ == "__main__":
    # Convert files in the merged_features directory
    data_dir = Path("data/merged_features")
    convert_pkl_to_csv(data_dir)
    logger.info("Conversion complete")
