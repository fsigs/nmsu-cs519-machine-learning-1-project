import sys
sys.path.append('../classes/')
import numpy as np
from AMLProject import AMLProject

if __name__ == "__main__":
  project = AMLProject()
  project.set_dataset()
  print(project.dataset.head(10))