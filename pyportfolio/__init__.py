import logging

# Add a NullHandler to the top-level logger for the library
# This prevents "No handler found" warnings if the consuming app doesn't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
