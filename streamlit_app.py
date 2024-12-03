import streamlit as st
from openai import OpenAI
import os
import tiktoken
import pandas as pd
from scipy import spatial
import ast

openai_api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

rag_df = pd.read_csv('ECC_RAG_embeddings.csv')
# convert embeddings from CSV str type back to list type
rag_df['embedding'] = rag_df['embedding'].apply(ast.literal_eval)
# Show title and description.
st.title("💬 Chat-ECC")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-4o model to generate specific answers related to the Elmira Country Club. "
    "This chat tool leverages specific information about the club that can be found [here](https://www.elmiracountryclub.com/). "
    "\n\nThe development of this chatbot is VERY much a work in progress. Please share any questions/comments/suggestions with [Mick Smith](csmith715@gmail.com) "
)

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# search function
def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the supporting markdown documentation from the Elmira Country Club website to answer questions about the golf club. These web pages specifically address ' \
                   'useful information such as Membership, Golf Information, Dining, and more. If the answer cannot be found in the pages, write "I could not find an answer." '
    question = f"\n\nQuestion: {query}"
    q_message = introduction
    for string in strings:
        next_article = f'\n\nWebsite page:\n"""\n{string}\n"""'
        if (
                num_tokens(q_message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            q_message += next_article
    return q_message + question

def ask(
        query: str,
        df: pd.DataFrame = rag_df,
        model: str = "gpt-4o",
        token_budget: int = 4096 - 500,
        print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    a_message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You are an expert at answering questions about the Elmira Country Club in Elmira, NY."},
        {"role": "user", "content": a_message},
    ]
    rag_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = rag_response.choices[0].message.content
    return response_message


# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("How can I join the Elmira Country Club?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        answer = ask(prompt)
        # stream = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            # response = st.write_stream(stream)
            response = st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": response})
