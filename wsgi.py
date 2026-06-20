import sys
import os

# Replace 'yourusername' with your PythonAnywhere username
project_home = '/home/costas/ScoreCast_v2'

if project_home not in sys.path:
    sys.path.insert(0, project_home)

from Scripts.WebApp.app import app as application
