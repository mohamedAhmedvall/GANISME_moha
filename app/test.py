# Vérifier que le repo existe et est accessible
import huggingface_hub as hfh

# Lister les fichiers du repo pour debug
files = hfh.list_repo_files("votre_repo_id")
print("Fichiers disponibles:", files)

# Si le repo est privé, utiliser un token
model_path = hf_hub_download(
    repo_id="movall/gan64",
    filename="dcgan_64.pt",
    token="HF_TOKEN"  # ou os.getenv("HF_TOKEN")
)