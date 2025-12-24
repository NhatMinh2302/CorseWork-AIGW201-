import pandas as pd
import joblib
import gradio as gr

from langchain.llms import Ollama
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory

from data_tools import load_data


model = joblib.load("models/sales_model.pkl")
scaler = joblib.load("models/scaler.pkl")



def data_analysis_tool(query: str) -> str:
    """Perform data analysis on Superstore dataset."""
    df = load_data()

    if "total sales last month" in query.lower():
        max_date = df["Order Date"].max()
        last_month = max_date.month - 1 if max_date.month > 1 else 12
        last_year = max_date.year if max_date.month > 1 else max_date.year - 1

        filtered = df[
            (df["Order Date"].dt.month == last_month)
            & (df["Order Date"].dt.year == last_year)
        ]
        total = filtered["Sales"].sum()
        return f"Total sales last month: {total:.2f}"

    elif "average profit" in query.lower():
        avg = df["Profit"].mean()
        return f"Average profit: {avg:.2f}"

    return "Query not supported."

def predict_sales_tool(input_str: str) -> str:
    """
    Predict sales.
    Input format: quantity, discount, month, year
    Example: 5, 0.2, 12, 2025
    """
    try:
        quantity, discount, month, year = map(float, input_str.split(","))
        input_data = [[quantity, discount, month, year]]
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        return f"Predicted sales: {prediction:.2f}"
    except Exception as e:
        return f"Invalid input format. Error: {e}"

tools = [
    Tool(
        name="Data Analysis",
        func=data_analysis_tool,
        description="Use for analyzing sales data, e.g. total sales last month, average profit."
    ),
    Tool(
        name="Sales Prediction",
        func=predict_sales_tool,
        description="Use for predicting sales. Input: quantity, discount, month, year."
    )
]



llm = Ollama(model="llama3")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)


def chat(message, history):
    response = agent.run(message)
    return response

demo = gr.ChatInterface(
    fn=chat,
    title="Supper AI",
    description=(
        "Make by Minh "
        "12-24-2025."
    ),
)

if __name__ == "__main__":
    demo.launch()
