"""Step-controlled Sakuradatasetservice input; edit the socket and training knobs."""

DATASETS = [{"source": "sakura_service", "socket": "/path/to/service-state/data.sock"}]
CAPTION = {"template": "multichoice"}
CAPTION_VARIANTS = 16
EPOCHS = 0
NUM_WORKERS = 0
LATENT_CACHE = "off"
TEXT_CACHE = "off"
# Conservative prewarm bound for future shards; increase for longer captions.
MAX_TEXT_TOKENS = 512
