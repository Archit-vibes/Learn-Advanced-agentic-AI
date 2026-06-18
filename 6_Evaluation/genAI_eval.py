from langsmith import Client
from langsmith import evaluate

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

#-----Intilize client

langsmith_client = Client()

#-----Define the Dataset

dataset = langsmith_client.create_dataset("Chatbot Eval Dataset #2")

langsmith_client.create_examples(
    dataset_id=dataset.id,
    examples = [
    {
        "inputs": {
            "question": "What is the capital of France?"
        },
        "outputs": {
            "answer": "Paris is the capital of France."
        }
    },
    {
        "inputs": {
            "question": "Who founded Tesla?"
        },
        "outputs": {
            "answer": "Tesla was founded by Martin Eberhard and Marc Tarpenning."
        }
    },
    {
        "inputs": {
            "question": "What is LangGraph?"
        },
        "outputs": {
            "answer": "LangGraph is a framework for building stateful, multi-agent workflows using graph-based orchestration."
        }
    },
    {
        "inputs": {
            "question": "Explain async functions in Python."
        },
        "outputs": {
            "answer": "An async function is declared with 'async def' and can pause execution using 'await' while waiting for asynchronous operations."
        }
    },
]
)

llm = ChatGroq(model="openai/gpt-oss-120b")

def correctness(
    inputs,
    outputs,
    reference_outputs,
):
    correctness_prompt = f"""
You are an expert evaluation judge.

Question:
{inputs["question"]}

Ground Truth Answer:
{reference_outputs["answer"]}

Predicted Answer:
{outputs["answer"]}

Determine whether the predicted answer is factually correct
with respect to the ground truth answer.

Respond with ONLY:
CORRECT
or
INCORRECT
"""

    response = llm.invoke(correctness_prompt)

    verdict = response.content.strip().upper()

    return {
        "key": "correctness",
        "score": 1 if verdict == "CORRECT" else 0,
    }

#-------Target Function

def target_functions(inputs:dict):
    response = llm.invoke(inputs["question"])

    return {"answer" : response.content}

#===============
# Run Evaluation
#===============

results = evaluate(
    target_functions,
    data = dataset.name,
    evaluators=[correctness],
    experiment_prefix="groq-chatbot-eval"
)

print("Evaluation Complete!")
print(results)