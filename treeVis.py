import pandas as pd
import numpy as np
from sklearn import tree

# Install pydotplus and graphviz
import pydotplus
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image

class vis(object):


    def __init__(self):
        pass

    def dtree_visual(self, treeClf):
        dot_data = StringIO()
        export_graphviz(treeClf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        return Image(graph.create_png())

    def __del__(self):
        pass




