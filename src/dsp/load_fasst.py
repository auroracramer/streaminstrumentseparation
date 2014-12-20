import os, sys

# Import the fasst package
fasst_python_dir = 'C:/Program Files/fasst 2.1.0/scripts/python'
if fasst_python_dir not in sys.path:
    sys.path.insert(0, fasst_python_dir)

# Remove from the global namespace so we don't import anything extra
del os
del sys
del fasst_python_dir

import fasst
