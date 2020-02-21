import json
import sys
from flask import Flask, request
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
import logging
import traceback
import time
logging.basicConfig(level=logging.INFO)


# Set up Flask app
app = Flask(__name__)

# ParlAI bot
BOT = {}
with open("hred_model_opt.json") as f:
	BOT["opt"] = json.load(f)
BOT["opt"]["task"] = "parlai.agents.local_human.local_human:LocalHumanAgent"
agent = create_agent(BOT.get("opt"), requireModelExists=True)
BOT["agent"] = agent
BOT["world"] = create_task(BOT.get("opt"), BOT.get("agent"))
logging.debug(f"Bot parlai settings: {BOT['opt']}")


def create_response(bot_reply, end_of_session=False):
	"""Base reply for Alexa"""
	response = {
  		"version": "1.0",
  		"response": {
    		"outputSpeech": {
      			"type": "PlainText",
      			"text": bot_reply,
    		},
    		"reprompt": {
      			"outputSpeech": {
        			"type": "PlainText",
        			"text": "Plain text string to speak",
        			"playBehavior": "REPLACE_ENQUEUED"             
      			}
    		},
    		"shouldEndSession": end_of_session
  		}
	}
	return response


def _interactive_running(opt, reply_text):
	"""Get response from bot"""
	reply = {"episode_done": False, "text": reply_text}
	BOT["agent"].observe(reply)
	bot_response = BOT["agent"].act()
	logging.info(bot_response)
	# bot_response = BOT["agent"].act().get("text", "")
	return bot_response


@app.route("/", methods=["POST"])
def interaction():
	# Get user request
	user_request = request.get_json().get("request")
	logging.debug(user_request)

	# Handle the requests
	if user_request.get("type") == "LaunchRequest":
		return create_response("Welcome to our chatbot, MalAIse. Please be nice to her. Happy chatting!")
	
	elif user_request.get("type") == "IntentRequest":
		intent = user_request.get("intent")
		logging.debug(intent)
		
		if intent.get("name") == "MimicIntent":
			return create_response(intent.get("slots").get("raw_input").get("value"))
		
		if intent.get("name") == "ChitChatIntent":
			user_utterance = intent.get("slots").get("raw_input").get("value")
			logging.debug(f"User: {user_utterance}")
			try:
				start = time.time()
				bot_response = _interactive_running(BOT.get("opt"), user_utterance)
				logging.debug(f"Bot: {bot_response}")
				end = time.time()
				logging.info(f"Model response took {end-start} seconds")
				return create_response(bot_response)
			except Exception as err:
				track = traceback.format_exc()
				logging.error(track)
				return create_response("Sorry there was a malfunction on my end.")

		if intent.get("name") == "AMAZON.StopIntent":
			BOT["agent"].reset()
			return create_response("Thanks for chatting. Goodbye!", end_of_session=True)		
	
		else:
			return create_response(f"Unknown intent '{intent.get('name')}'")

	elif user_request.get("type") == "SessionEndedRequest":
		BOT["agent"].reset()
		return create_response("Session ended.", end_of_session=True)

	else:
		return create_response(f"Unknown request type '{user_request.get('type')}'")


if __name__ == "__main__":
	# Start server
	app.run()


