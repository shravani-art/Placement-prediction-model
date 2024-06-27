import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

placement_df=pd.read_csv("Campus_Selection.csv")
# placement_df
placement_df_required=placement_df.drop("sl_no",axis=1)
# placement_df_required
