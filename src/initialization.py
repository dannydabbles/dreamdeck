import os
import logging

import chainlit as cl
import chainlit.data as cl_data
from chainlit.config import config, DEFAULT_HOST
from chainlit.element import ElementDict
from chainlit.types import ThreadDict
from chainlit.element import Image as CLImage

from chainlit import Message as CLMessage
from chainlit.element import Text as CLText

from starlette.staticfiles import StaticFiles
from chainlit.server import app as chainlit_app

from config import (
    DATABASE_URL,
    KNOWLEDGE_DIRECTORY,
)

#from image_hosting import FSStorageClient
#from data_layer import CustomDataLayer, init_db

# Initialize logging
cl.logger = logging.getLogger("chainlit")

# Initialize the storage client
#fs_storage_client = FSStorageClient(
#    storage_path=os.path.join(os.getcwd(), "public", "storage"),
#    url_path=os.path.join("public", "storage")
#)

# Mount the static files directory to serve images
#chainlit_app.mount("/storage", StaticFiles(directory=os.path.join("public", "storage")), name="storage")

# Initialize the custom data layer with the storage client
#cl_data._data_layer = CustomDataLayer(conninfo=DATABASE_URL, storage_provider=[fs_storage_client])

# Initialize the database
#init_db()
