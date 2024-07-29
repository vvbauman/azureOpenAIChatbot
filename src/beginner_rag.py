import os
from dotenv import load_dotenv
from openai import AzureOpenAI

def clear_env_vars():
    for var in ["OAI_ENDPOINT", "OAI_KEY", "OAI_DEPLOYMENT", "API_VERSION", "SEARCH_ENDPOINT", "SEARCH_KEY", "SEARCH_INDEX", "MODEL_NAME"]:
        os.environ.pop(var)
    return

def get_config():
    """
    Loads environment variables, 
    creates the data source to include in an AzureOpenAI client object's chat.completions.create method, 
    creates an AzureOpenAI client using the environment variables

    Returns
    ----------
    oai_client : AzureOpenAI client object, defined using environment variables in .env
    oai_deployment : str. Name of AzureOpenAI deployment (i.e., foundation model) used in oai_client
    data_source_config : dict. Data source/AI Search specifications, used with oai_client to ground responses 
    """
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

    # specify AI Search resource that has data we want to use
    # see https://learn.microsoft.com/en-us/azure/ai-services/openai/references/azure-search?tabs=python for all parameters
    data_source_config = {"data_sources": [{
            "type": "azure_search", 
             "parameters": { 
                 "endpoint":search_endpoint, 
                 "index_name":search_index,
                 "authentication": {
                     "type":"api_key",
                     "key":search_key,
                 },
             },
    }],
    }

    # create Azure OpenAI client object
    oai_client = AzureOpenAI(
        azure_endpoint = oai_endpoint, 
        api_key = oai_key,
        api_version = api_version, 
    )
    return oai_client, oai_deployment, data_source_config

# benefits of using code below:
# good starting point to create OpenAI client objects and use the chat.completions.create method - there are many parameters you can provide to this method to improve chat responses for your use case
# good way to get familiar with maintaining chat history
# good way to experiment with system message
# good way to understand environment variables

# problems with approach below:
# not token-efficient - conversation history is maintained by appending all conversation turns
# not search-efficient - keywords for azure search are created from entire conversation history
def main(): 
    # Create Azure OpenAI client, define constants to be used in all calls to the client
    oai_client, azure_oai_deployment, data_source_config = get_config()
    system_message = """Assistant helps the company employees with their healthcare plan questions, and questions about the employee handbook. Be brief in your answers.
        Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. 
        Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
        Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, for example [info1.txt]. 
        Don't combine sources, list each source separately, for example [info1.txt][info2.pdf].
        """
    temperature = 0.3 # response creativity (0-2, 0 being entirely factual and literal)
    max_tokens = 1000 # repsonse token limit. 1 token ~= 4 characters
    max_questions = 3 # max turns the conversation has before program exits
    # start creating messages to be sent to Azure OpenAI client
    message_list = [
        {"role": "system", "content": system_message},
        ]

    print("Welcome to the Contoso help chatbot!")

    q= 0
    while q < max_questions:
        # Get prompt from user
        text = input('\nEnter a question:\n')

        # Send request to Azure OpenAI model
        print("...Sending the following request to Azure OpenAI endpoint...")
        print("Request: " + text + "\n")

        # append user input to message_list
        message_list.append(
            {"role": "user", "content": text}
        )

        # send messages to OpenAI client
        # see https://github.com/openai/openai-python/blob/main/src/openai/resources/completions.py for all args
        response = oai_client.chat.completions.create(
            model = azure_oai_deployment,
            temperature = temperature,
            max_tokens = max_tokens,
            messages = message_list,
            extra_body = data_source_config, # Azure AI Search details
        )
        text_reply = response.choices[0].message.content + "\n"

        # Print response
        print("Response: " + text_reply + "\n")
        # add response to message_list, to maintain conversation history
        message_list.append(
            {"role": "assistant", "content": text_reply}
        )
        q += 1
    
    print("Thanks for chatting! Goodbye")

if __name__ == '__main__': 
    main()

# TODO: update improved_rag.py based on changes in this beginner_rag.py

