import wandb
import pandas as pd
run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-xgboost-models")

# create a wandb Artifact for each meaningful step
artifact = wandb.Artifact(
"wpg_v_wsh_2017021065", 
type="dataset"
)

# add data
df = pd.read_csv("../dataset/complex_engineered/2017/game_2017021065.csv")
my_table = wandb.Table(dataframe=df)
artifact.add(my_table, "wpg_v_wsh_2017021065")
run.log_artifact(artifact)
