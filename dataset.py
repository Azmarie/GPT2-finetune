import pandas as pd
import json

def make_dataset(path_to_train_data="data/train_data.json", path_to_validation_data="data/validate_data.json"):
  f_train = open(path_to_train_data)
  train_data = json.load(f_train)
  f_train.close()
  # print(len(train_data))
  
  f_validate = open(path_to_validation_data)
  validate_data = json.load(f_validate)
  f_validate.close()
  # print(len(validate_data))

  train_contexted = []
  train_data = train_data
  
  for i in range(len(train_data)):
    row = []
    row.append(train_data[i][1])
    row.append(train_data[i][0])
    train_contexted.append(row)  

  validate_contexted = []

  for i in range(len(validate_data)):
    row = []
    row.append(validate_data[i][1])
    row.append(validate_data[i][0])
    validate_contexted.append(row)  

  columns = ['response', 'context'] 
  columns = columns + ['context/'+str(i) for i in range(0)]

  trn_df = pd.DataFrame.from_records(train_contexted, columns=columns)
  val_df = pd.DataFrame.from_records(validate_contexted, columns=columns)

  return trn_df,val_df