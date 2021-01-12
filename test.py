# Below are code blocks from `ipynb`

test_chatbot = []

for i in range(len(test_query)):
  tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
  model = AutoModelWithLMHead.from_pretrained('output-small')
  # append the new user input tokens to the chat history
  bot_input_ids = tokenizer.encode(test_query[i] + tokenizer.eos_token, return_tensors='pt')
  print("User: {} \n".format(test_query[i]))

  # generated a response while limiting the total chat history to 1000 tokens, 
  chat_history_ids = model.generate(
      bot_input_ids, max_length=100,
      pad_token_id=tokenizer.eos_token_id,  
      no_repeat_ngram_size=3,       
      do_sample=True, 
      top_k=10, 
      top_p=0.7,
      temperature = 0.8
  )

  # pretty print last ouput tokens from bot
  print("Chatbot: {} \n\n".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
  test_chatbot.append(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

print(len(test_chatbot))
