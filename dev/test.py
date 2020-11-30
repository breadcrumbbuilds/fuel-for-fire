import sys, os
sys.path.append(os.curdir)
from Utils.Misc import get_working_directories


directories = get_working_directories("debug/files", ['output', 'model'])

print(directories)