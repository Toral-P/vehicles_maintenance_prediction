from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="vehicles_predictive_maintenance/deployment",
    repo_id="toriaiml/Vehicles-Predictive-Maintenance",
    repo_type="space",
)
