# In ~/lumiere_semantique/backend/run_server.sh
#!/bin/bash
#
# This script is the standard way to run the Lumière Sémantique development server.
# It ensures the server always starts on the correct port (8002).
#
# To run:
# 1. Make it executable: chmod +x run_server.sh
# 2. Execute it: ./run_server.sh

echo "Starting Lumière Sémantique development server on http://127.0.0.1:8002/"
python manage.py runserver 8002
