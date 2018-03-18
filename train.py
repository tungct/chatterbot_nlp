from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
import logging
import re
# logging.basicConfig(level=logging.INFO)

# train file
f = open('./data/fb.txt','r')
train_data = []

# read line to conversation with Q and A
for line in f:
    m = re.search('(Q:|A:)?(.+)', line)
    if m:
        train_data.append(m.groups()[1])

# create chatbot with name
chatbot = ChatBot("weatherHanoi",
                     storage_adapter="chatterbot.storage.SQLStorageAdapter",
                     input_adapter="chatterbot.input.TerminalAdapter",
                     output_adapter="chatterbot.output.TerminalAdapter",
                     )

chatbot.set_trainer(ListTrainer)

chatbot.train(train_data)
