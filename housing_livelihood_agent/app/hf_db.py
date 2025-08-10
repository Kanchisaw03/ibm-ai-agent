import logging
from datetime import datetime
from typing import Dict, Any

from datasets import Dataset, DatasetDict, load_dataset, Features, Value, concatenate_datasets
from huggingface_hub import HfFolder
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)

def setup_hf_hub() -> None:
    """Saves the Hugging Face token from settings to the local machine."""
    if settings.HF_TOKEN:
        HfFolder.save_token(settings.HF_TOKEN)
    else:
        logger.warning("HF_TOKEN not set. Hugging Face operations will fail.")

# Updated schema for the Hugging Face dataset
DATASET_FEATURES = Features({
    'vendor_id': Value('string'),
    'recommended_zone': Value('string'),
    'rationale': Value('string'),
    'ml_score': Value('float64'),
    'timestamp': Value('string'),
    'map_url': Value('string'),
})

def _get_dataset() -> DatasetDict:
    """Loads or initializes the Hugging Face dataset."""
    try:
        dataset = load_dataset(settings.HF_DATASET_REPO, token=settings.HF_TOKEN)
        logger.info(f"Successfully loaded existing dataset from {settings.HF_DATASET_REPO}.")
        return dataset
    except Exception:
        logger.warning(
            f"Dataset not found at {settings.HF_DATASET_REPO}. "
            "Initializing a new empty dataset."
        )
        empty_ds = Dataset.from_dict({
            'vendor_id': [], 'recommended_zone': [], 'rationale': [],
            'ml_score': [], 'timestamp': [], 'map_url': [],
        }, features=DATASET_FEATURES)
        return DatasetDict({'train': empty_ds})

def _flatten_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens the plan dictionary for saving to the dataset."""
    timestamp = plan.get("timestamp")
    return {
        "vendor_id": plan.get("vendor_id"),
        "recommended_zone": plan.get("recommended_zone"),
        "rationale": plan.get("rationale"),
        "ml_score": plan.get("ml_score"),
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
        "map_url": plan.get("map_url", "")
    }

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def add_plan_to_hf_dataset(plan_data: Dict[str, Any]) -> None:
    """
    Adds the relocation plan to the Hugging Face Dataset.
    """
    try:
        logger.info("Adding plan to Hugging Face Dataset.")
        setup_hf_hub()
        
        dataset = _get_dataset()

        flattened_plan = _flatten_plan(plan_data)
        new_data_ds = Dataset.from_dict({k: [v] for k, v in flattened_plan.items()}, features=DATASET_FEATURES)

        if 'train' in dataset and len(dataset['train']) > 0:
            dataset['train'] = concatenate_datasets([dataset['train'], new_data_ds])
        else:
            dataset['train'] = new_data_ds
        
        dataset.push_to_hub(
            repo_id=settings.HF_DATASET_REPO,
            token=settings.HF_TOKEN,
            private=True,
        )
        logger.info(f"Successfully pushed updated dataset to {settings.HF_DATASET_REPO}")

    except RetryError as e:
        logger.error(f"Operation failed after multiple retries: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while updating the HF Dataset: {e}")
        raise
