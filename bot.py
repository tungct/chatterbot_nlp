from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import logging

# logging.basicConfig(level=logging.INFO)

chatterbot = ChatBot("weatherHanoi",
                     storage_adapter="chatterbot.storage.SQLStorageAdapter",
                     input_adapter="chatterbot.input.TerminalAdapter",
                     output_adapter="chatterbot.output.TerminalAdapter",
                     )

print("Type something to begin...")

# Get a response to an input statement
while True:
    try:
        # We pass None to this method because the parameter
        # is not used by the TerminalAdapter
        bot_input = chatterbot.get_response(None)

    # Press ctrl-c or ctrl-d on the keyboard to exit
    except (KeyboardInterrupt, EOFError, SystemExit):
        break
