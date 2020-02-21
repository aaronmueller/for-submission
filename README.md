Names: Lisa Li, Aaron Mueller, Alexandra DeLucia

Emails: {xli150, amueller, aadelucia}@jhu.edu

-------------------------------------------------------------

# Reproduction Instructions

0. Copy the files from this repository into your installation of ParlAI. A few files will be replaced, so we recommend creating backups beforehand.

1. Now, we need to preprocess the Movie Triples dataset. 
Download the Movie Triple Data to ParlAI/data, unzip it and rename the file to MovieTriples_Dataset.tar. Inside the MovieTriple_Dataset file, we also need to rename the triple txt files to {train, valid, test}.txt, respectively. Our code will take care of the rest.

2.  **Training**: 
We provide training scripts for easy reproduction of our training setup. We roughly follow the hyperparameters presented in the original HRED paper---with a few modifications, given the constraints of the assignment. Note that this script is set up to work on the CLSP grid, and you will most certainly need to make some modifications to make it work for your machine or installation.
```
train.sh
```
It (i) sources a ParlAI-enabled conda environment w/ CUDA-enabled PyTorch, (2) sets the proper environment variables, and (3) calls the training function with all the appropriate hyperparameters. We have also included `train_2.sh`, which is similar to `train.sh` but with some (unstable) multi-threading behaviors. We recommend using `train.sh`. 

3. **Evaluation**:
In the ParlAI/ directory, we have provided an evaluation script named `ppl_eval.sh`. Run this to replicate our evaluation conditions. It will produce an output file named `s` containing eval information on test.:
Note that this script evaluates on internal:dailydialog. Because we have replaced dailydialog with our MovieTriples dataset in our internal implementations, this will actually evaluate on the validation and test sets of MovieTriples.


4. **Interactive**:
From the ParlAI/ directory, run the following command:
```
python examples/interactive.py -mf parlai_internal/zoo/movie_hred/hred_model.ckpt.checkpoint -m internal:hred
```

5. **Integrating with Alexa**:
For Alexa integration we used `ngrok`, a free port forwarding service, and `Flask`, a lightweight python server package, to host a custom HTTPS endpoint for our chatbot. 

The integration files are in `alexa_integration`. You can start the Alexa skill server with
```
./start_skill_server.sh
```
This will host the skill on port 2020. And then to forward the port to 443 (the HTTPS port needed for Alexa), start ngrok in another terminal with 
```
ngrok http https://localhost:2020
```

And on the Alexa Development Console side, make a new skill with a custom HTTPS endpoint, using the URL assigned by ngrok. For the skill's JSON Editor, use the setup in `alexa_skill.json`. You can test the skill with the Development Console Test tool.


This integration can be used with any ParlAI model. Just create a file similar to `hred_model_opt.json` and point to that file in `alexa_server.py`.

Relevant Resources:  

* [ngrok](https://ngrok.com/docs)
* [flask](https://flask.palletsprojects.com/en/1.1.x/) 
* [Alexa response / request format information](https://developer.amazon.com/en-US/docs/alexa/custom-skills/request-and-response-json-reference.html)


# Qualitative Evaluations
Similar to other groups, we note that our model has low variance but very high bias. In other words, the model always outputs the same response regardless of what the user input is---and this response is useless in most situations.

The chatbot responds with "robotic legalistic robotic", followed by a long and repetitive series of "nehru" tokens. We tried a variety of inputs, but this seems to be the only response that the bot is capable of producing. It is unclear why it is unable to output more sensible responses, though this does yield plenty of inspiration for different types of evaluation metrics than have previously been proposed.

For example, consider BLEU: it has a brevity penalty and is essentially just a modified form of n-gram precision. With our chatbot's uniformly long response, it will never be subject to the brevity penalty and may sometimes demonstrate very small n-gram overlap in certain specialized domains. This is undesirable, since humans are easily able to tell that the output is degenerate regardless of its length. Perhaps we could define a new metric that encourages chatbot responses to be similar in length to a reference response, or a metric which discourages repetitive sequences of the same token such that outputs are more naturalistic. Or, perhaps we could go a step further and train a language model on similar-domain data, then obtain perplexities on our chatbot's outputs as a rough measure of how naturalistic the output is.

There are a variety of metrics that could be used to qualitatively and automatically judge the performance of a chatbot system based on the flaws of our current system, but ultimately, we do not need any of these to see that it does not produce naturalistic responses. This leads to our next section on our model's current issues and ideas for future improvement.

# Quantitative Evaluations
Our training perplexity was 205.8 on MovieTriples. Validation perplexity was 378.1, and test perplexity was 462.4. These are likely the most informative metrics, since we can compare these with the HRED paper's relatively low perplexities.

BLEU scores were less informative; they were uniformly low due to the repetitive output. Our validation and test BLEU were less than 0.01, most of which likely comes from punctuation.

# Issues and Potential Improvements
1. From our qualitative evaluation, our trained model is outputting with little variance -- essentially, it outputs the same sentence despite the various inputs that we tried.

2. Some easy improvement ideas include training the system for a longer period of time; carefully tuning and searching over the hyperparameters; or initializing the model with some pre-trained language model and then fine-tuning. We could initialize our model with GPT-2 to obtain a language-modeling-aware starting point, which would certainly aid the fluency (even if not the acceptability) of our output.

3. At decoding time, we could use the mutual information objective (A Diversity-Promoting Objective Function for Neural Conversation Models) or the Nucleus sampling techniques (The Curious Case of Neural Text Degeneration) to promote more diversity in the model's output. This could likely improve on the low-variance but high-bias system we currently have.


# Find our Model
If you want to use our trained models, they are available on the CLSP grid. Look in `/export/b10/amueller/discourse/hw3/discourse-hw3/parlai_internal/zoo/movie_hred/hred_model.ckpt.checkpoint`. We also have other models in this folder from other experiments (though we do not recommend them since the aforementioned path is to our best model).
