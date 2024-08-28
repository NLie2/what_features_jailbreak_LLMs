from datetime import datetime
import os
import pickle

def save_with_timestamp(model_name, dataset_name, save=False, df=None, add_info=""):
    now = datetime.now().strftime("%d%m%Y%H:%M:%S")

    model_name = model_name.replace("/", "")
    dataset_name = dataset_name.replace("/", "")

    save_dir = "/data/nathalie_maria_kirch/ERA_Fellowship/datasets"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f"{save_dir}/{dataset_name}_{model_name}_{add_info}_{now}.pkl"
    
    if save: 
      with open(save_path, 'wb') as f:
          pickle.dump(df, f)

    return save_path