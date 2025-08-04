from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder,GroupNormalizer, TemporalFusionTransformer
from pytorch_lightning import Trainer
# import torch
import pytorch_lightning
import pytorch_forecasting
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# import plotly.express as px
from pytorch_forecasting.metrics import QuantileLoss
from tft import PollutionTFT

pollution_cleaned = pd.read_csv("cleaned_pollution_data.csv")
pollution_cleaned["Date"] = pd.to_datetime(pollution_cleaned["Date"])

df_tft = pollution_cleaned[pollution_cleaned['City'] == "Los Angeles"].copy()
df_tft = df_tft.dropna()

df_tft["City"] = df_tft["City"].astype(str)
df_tft["time_idx"] = (df_tft["Date"] - df_tft["Date"].min()).dt.days

target_columns = ["O3 Mean", "CO Mean", "SO2 Mean"]
df_long = df_tft.melt(
    id_vars=["time_idx", "City", "Month", "DayOfWeek", "IsWeekend", "IsWedThur",
             "O3 Mean_lag1", "CO Mean_lag1", "SO2 Mean_lag1", "NO2 Mean_lag1", "Pollution_Avg"],
    value_vars=target_columns,
    var_name="target_variable",
    value_name="target"
)
df_long = df_long.dropna(subset=["target"])

max_prediction_length = 7
max_encoder_length = 30
training_cutoff = df_tft["time_idx"].max() - max_prediction_length

tft_dataset = TimeSeriesDataSet(
    df_long[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="target",
    group_ids=["City", "target_variable"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["City", "target_variable"],
    time_varying_known_reals=["time_idx", "Month", "DayOfWeek", "IsWeekend", "IsWedThur"],
    time_varying_unknown_reals=["target", "Pollution_Avg", "O3 Mean_lag1", "CO Mean_lag1", "SO2 Mean_lag1", "NO2 Mean_lag1"],
    target_normalizer=GroupNormalizer(groups=["City", "target_variable"], transformation=None),
    allow_missing_timesteps=True
)

val_dataset = TimeSeriesDataSet.from_dataset(
    tft_dataset,
    df_long,
    predict=True,
    stop_randomization = True,
)

tft = TemporalFusionTransformer.from_dataset(
    tft_dataset,
    learning_rate = 0.3,
    hidden_size = 16,
    attention_head_size = 1,
    dropout = 0.1,
    hidden_continuous_size = 8,
    output_size = 7,
    loss=QuantileLoss(),
    log_interval=-1,
    reduce_on_plateau_patience=4,
)

print(type(tft))
print(isinstance(tft, pytorch_lightning.LightningModule))
print(TemporalFusionTransformer.__module__)

trainer = Trainer(
    max_epochs=50,
    accelerator="cpu",
    devices=1,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    log_every_n_steps=10,

)

train_dataloader = tft_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64*10, num_workers=0)

trainer.fit(
    tft,
    train_dataloaders =train_dataloader,
    val_dataloaders=val_dataloader,
)