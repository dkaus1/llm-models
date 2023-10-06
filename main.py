from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from flask import Flask, request, make_response

app = Flask(__name__)


B_INST,E_INST = "[INST]","[/INST]"
B_SYS, E_SYS ="<<SYS>>\n", "\n<<SYS>>\n\n"

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="codellama-13b-instruct.Q4_K_M.gguf",
    n_ctx =2048,
    max_tokens=5000,
    n_gpu_layers=1,
    n_batch=512,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=False,
    temperature=0.2
)

# check if the app is running
@app.route('/status', methods=['GET'])
def live_check():
    return "Model API is running"


def prompt_builder(question, system_prompt):
    instruction = "User: " + question
    sys_prompt = B_SYS + system_prompt + E_SYS
    prompt_template = B_INST + sys_prompt +instruction +E_INST
    return prompt_template


@app.route('/predictions', methods=['POST'])
def prediction():
    body = request.get_json(silent=True,force=True)
    if not body or "question" not in body.keys() or "system_prompt" not in body.keys():
        return make_response('Bad Request',400)
    
    question = body["question"]
    system_prompt= body["system_prompt"]
    message = prompt_builder(question, system_prompt)
    return llm(message)


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)