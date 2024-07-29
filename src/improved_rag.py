# adapted from chatreadretrieveread.py from https://github.com/Azure-Samples/azure-search-openai-demo
# linked repo deploys a chatgpt-like web-app and has a lot of built-in features
# goal of this script is to provide starter code on using some of the libraries in this linked repo (stepping stone to chatreadretrieveread.py)
# steps we'll accomplish:
# 1) facilitate a 3-turn conversation between chatbot and user
# 2) for each user prompt, create an optimized search query for AI Search
# 3) retrieve relevant docs from AI Search using the optimized search query
# 4) create a content-specific answer and return to user using the search result and chat history
# TODO - add "re-write question" stage?

import os
import json
from dotenv import load_dotenv
from dataclasses import dataclass
from openai import AzureOpenAI
from typing import List, Optional, Any
# from azure.search.documents.aio import SearchClient
from azure.search.documents import SearchClient
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionToolParam,
)
from openai_messages_token_helper import build_messages, get_token_limit # source: https://github.com/pamelafox/openai-messages-token-helper

@dataclass
class Doc:
    """Class for keeping track of docs retrieved from AI Search"""
    content: Optional[str]
    score: Optional[float] = None
    highlight: Optional[list[Any]] = None

def clear_env_vars():
    for var in ["OAI_ENDPOINT", "OAI_KEY", "OAI_DEPLOYMENT", "API_VERSION", "SEARCH_ENDPOINT", "SEARCH_KEY", "SEARCH_INDEX"]: #, "MODEL_NAME"]:
        os.environ.pop(var)
    return

def get_config():
    clear_env_vars()
    load_dotenv()
    # get environment variables (keys, endpoints, etc.)
    oai_endpoint = os.getenv("OAI_ENDPOINT")
    oai_key = os.getenv("OAI_KEY")
    oai_deployment = os.getenv("OAI_DEPLOYMENT")
    api_version = os.getenv("API_VERSION")
    search_endpoint = os.getenv("SEARCH_ENDPOINT")
    search_key = os.getenv("SEARCH_KEY")
    search_index = os.getenv("SEARCH_INDEX")
    
    # create Azure OpenAI client object
    oai_client = AzureOpenAI(
        azure_endpoint = oai_endpoint, 
        api_key = oai_key,
        api_version = api_version, 
    )

    from azure.core.credentials import AzureKeyCredential
    # create AI Search client object
    search_client = SearchClient(
        endpoint=search_endpoint, 
        index_name=search_index, 
        credential=AzureKeyCredential(search_key), #search_key,
        )

    return oai_client, search_client

def get_search_query(chat_completion: ChatCompletion, user_query: str):
    """
    Get optimized search query
    """
    response_msg = chat_completion.choices[0].message
    if response_msg.tool_calls: # this means that a function defined in our tools was called
        for tool in response_msg.tool_calls:
            if tool.type != "function":
                    continue
            function = tool.function
            if function.name == "search_sources": # get result from the search_sources function, if it was called
                arg = json.loads(function.arguments)
                search_query = arg.get("search_query", 0)
                if search_query != 0:
                    return search_query
    elif query_text := response_msg.content:
        if query_text.strip() != 0:
            return query_text
    return user_query


def main():
    # get OpenAI client and specify some chat completion parameters, same as before
    oai_client, search_client = get_config()
    system_message = """You are an assistant that summarizes document highlights retrieved from documents using Azure AI Search. You will receive the search query in double asterisks, for example, **eye exams**.
        You should start all responses with "this is what I know about **search query**". For example, for search query **eye exams**, your response will start as "This is what I know about eye exams".
        Your responses should be 2-3 sentences and should include all key details without adding external information or assumptions. 
        """
    temperature = 0.3 # response creativity (0-2, 0 being entirely factual and literal)
    max_tokens = 1000 # repsonse token limit. 1 token ~= 4 characters
    max_questions = 3 # max turns the conversation has before program exits

    # get the token limit for the model we've deployed - see https://github.com/pamelafox/openai-messages-token-helper/blob/main/src/openai_messages_token_helper/model_helper.py
    deployment_name = os.getenv("OAI_DEPLOYMENT")
    model_name = os.getenv("MODEL_NAME")
    model_token_limit = get_token_limit(model=model_name)

    # create the prompt we'll use to create the optimized search query
    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
        You have access to Azure AI Search index with some documents.
        Generate a search query based on the conversation and the new question.
        Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
        Do not include any text inside [] or <<>> in the search query terms.
        Do not include any special characters like '+'.
        If you cannot generate a search query, return just the number 0.
        """   
    
    query_resp_token_limit = 100 # max tokens to create optimized search query
    
    # define tools used to build messages to get optimized search query - see https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
    # this has good examples on how tools are used too: https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
    tools : List[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": {
                        "name": "search_sources",
                        "description": "Retrieve sources from the Azure AI Search index",
                        "parameters": {
                                "type": "object",
                                "properties": {
                                        "search_query": {
                                                "type": "string",
                                                "description": "Query string to retrieve documents from azure search using simple query search eg: 'Health care plan'",
                                                        }
                                                },
                                                "required": ["search_query"],
                                        },
                }
            }
    ]

    q = 0
    while q < max_questions: # STEP 1) Facilitate a 3-turn conversation

        # Get prompt from user
        text = input('\nEnter a question:\n')

        # STEP 2) Create an optimized search query from the user input
        # build messages to send to model to get search query - see https://github.com/pamelafox/openai-messages-token-helper/blob/main/src/openai_messages_token_helper/message_builder.py
        query_messages = build_messages(
            model = model_name, # need openAI-friendly name here
            system_prompt = query_prompt_template,
            tools = tools,
            past_messages = [] if q == 0 else messages[:-1], # TODO: figure out how this works for the first turn
            # user_request_query = "Generate search query for: " + text, # not a valid argument for this method
            new_user_content = "Generate search query for: " + text,
            max_tokens = model_token_limit - query_resp_token_limit,
        )       
        # send the messages to the Azure OpenAI client to create the optimized search query
        chat_completion : ChatCompletion = oai_client.chat.completions.create(
            messages = query_messages,
            model = deployment_name,
            temperature = 0, # minimize creativity for search query creation
            max_tokens = query_resp_token_limit,
            n = 1,
            tools = tools,
        )

        query_text = get_search_query(chat_completion=chat_completion, user_query=text) #"eye_exam"

        # STEP 3) Retrieve documents from AI Search using the optimized query
        search_results = search_client.search(
            search_text=query_text, # optimized search query we created before
            top=2, # number of search results to retrieve
            query_type="simple",
            highlight_fields="content",
            )

        docs = []
        for page in search_results.by_page():
            for doc in page:
                docs.append(
                    Doc(
                        content=doc.get("content"),
                        score=doc.get("@search.score"),
                        highlight=doc.get("@search.highlights")
                        # need to retrieve citation too
                        # TODO: when creating search resource, need to enable "metadata_storage_name" to be able to retrieve citations
                    )
                )

        # STEP 4) Create content-specific answer using the search results and chat history
        # create messages to send to OpenAI model to generate the response
        messages = build_messages(
            model=model_name,
            system_prompt=system_message,
            past_messages= [] if q == 0 else messages[:-1],
            new_user_content=f"**{query_text}**" + " ".join([e for d in docs for e in d.highlight["content"]]),
            max_tokens=model_token_limit - 1024,
        )
    
        # create and print response
        chat_reply = oai_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )
        display_chat = chat_reply.choices[0].message.content + "\n"
        print("Response: " + display_chat + "\n")

        # # TODO: double-check the object type of chat_reply and whether it makes sense to append to messages
        # messages.append(chat_reply)
        messages = []
        messages.append(messages)
        q += 1
    
    print("Thanks for chatting! Goodbye")

if __name__ == '__main__': 
    main()





        

