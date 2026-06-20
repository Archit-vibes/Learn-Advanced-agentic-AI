from langchain_litellm import ChatLiteLLM , ChatLiteLLMRouter

from litellm import Router
from litellm.cost_calculator import completion_cost

import time

from dotenv import load_dotenv

load_dotenv()
#===========
# Router Conifg
#===========

model_list = [
    {
        "model_name" : "code-model",
        "litellm_params" : {
            "model" : "openai/gpt-oss-120b"
        }
    } ,
    {
        "model_name" : "code-model",
        "litellm_params" : {
            "model" : "groq/llama-3.3-70b-versatile"
        }
    } ,

    {
        "model_name" : "text-model" ,
        "litellm_params" : {
            "model" : "groq/llama-3.3-70b-versatile"
        }
    } ,

    {
        "model_name" : "fast-model" , 
        "litellm_params" : {
            "model" : "groq/llama-3.1-8b-instant"
        }
    } , {
        "model_name" : "fast-model" , 
        "litellm_params" : {
            "model" : "groq/llama-3.3-70b-versatile"
        }
    }
]

router = Router(model_list=model_list , set_verbose=True)

#==============
# Classifier
#==============

classifier = ChatLiteLLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0
)

def classify_task(query:str):
    reponse = classifier.invoke(
        f"Classify the following query into EXACTLY one word: 'code', 'summary', or 'general'. Query: {query}"
    )

    result = reponse.content.strip().lower()

    if "code" in result:
        return "code"
    
    elif "summary" in result:
        return "summary"
    
    return "general"

    
def smart_chat(query:str):

    task = classify_task(query)

    router_map = {
        "code" : "code-model",
        "summary" : "text-model",
        "general" : "fast-model"
    }

    selected_route = router_map[task]

    start = time.time()

    response = router.completion(
        model=selected_route,
        messages=[
            {
                "role" : 'user',
                "content" : query
            }
        ]
    )

    latency = round(time.time() - start , 2)

    usage = response.usage

    prompt_tokens = usage.prompt_tokens 
    completion_tokens = usage.completion_tokens 
    total_tokens = usage.total_tokens

    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        print("Cost calculation failed:", e)
        cost = 0

    return { 
        "task": task, 
            "model_used": response.model, 
            "answer": response.choices[0].message.content, 
            "latency": latency, "prompt_tokens": prompt_tokens, 
            "completion_tokens": completion_tokens, 
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 8)

            }

queries = [
    "Write a Python function to compute Fibonacci numbers.",
    "Summarize the importance of attention mechanism in 2 sentences.",
    "Tell me a fun fact about elephants."
]

for q in queries:

    print("=" * 70)
    print("❓ Q:", q)

    result = smart_chat(q)

    print(f"🏷️  Task:    {result['task']}")
    print(f"🤖 Model:    {result['model_used']}")
    print(f"⏱️  Latency: {result['latency']}s")
    print(f"💰 Cost:    ${result['cost_usd']}")
