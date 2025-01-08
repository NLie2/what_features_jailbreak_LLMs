def rate_dataset(df, request_column, response_column, output_path):
  # Load the HarmBench model
  model_name = "cais/HarmBench-Llama-2-13b-cls"
  autograder = AutoModelForCausalLM.from_pretrained(model_name)
  autograder.to('cuda')
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, truncation_side="left", padding_side="left")
  
  # Apply the grading function and add the ratings column
  df[f"rating_{response_column}"] = df.apply(lambda row: grade_with_HF(row[request_column], [row[response_column]], autograder, tokenizer, verbose = True)[0], axis=1)
  print("rated")


  # Save the dataframe to the output path
  df.to_pickle(output_path)
  print(output_path)
