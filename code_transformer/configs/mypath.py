import os
PROJECT_ROOT_DIR = os.path.abspath(os.curdir)

db_root =  '/storage/dsinod/Dimitri/datasets/atrc/' # CHANGE THIS

db_names = {'PASCALContext': 'PASCALContext'}
db_paths = {}
for database, db_pa in db_names.items():
    db_paths[database] = os.path.join(db_root, db_pa)