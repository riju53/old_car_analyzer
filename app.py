import streamlit as st
import requests
import sqlite3

from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tathagata The Used Car Advisor and Price Predictor.",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# API CONFIGURATION
# ============================================================

api_url = "https://car-prediction-fastapi-2.onrender.com/predict"


# ============================================================
# GROQ MODEL
# ============================================================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

model = ChatGroq(
    model_name="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2
)


# ============================================================
# SQLITE CHECKPOINTER
# ============================================================

conn = sqlite3.connect(
    "langgraph.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)


# ============================================================
# TITLE
# ============================================================

st.title("🚗 Tathagata The Used Car Advisor and Price Predictor.")

st.sidebar.title("🚗 Car Price Prediction and Advisor")


# ============================================================
# SIDEBAR INPUTS
# ============================================================

Brand = st.sidebar.selectbox(
    "Brand",
    [
        'Hyundai',
        'Volkswagen',
        'Toyota',
        'Honda',
        'Maruti',
        'Mahindra',
        'Tata',
        'Kia'
    ]
)


Car_Age = st.sidebar.number_input(
    "Car Age",
    min_value=0,
    max_value=12,
    value=9
)


Kilometers_Driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    value=31788
)


Engine_CC = st.sidebar.number_input(
    "Engine_CC",
    min_value=0,
    value=1498
)


Mileage_KMPL = st.sidebar.number_input(
    "Mileage_KMPL",
    min_value=0.0,
    value=20.7
)


Fuel_Type = st.sidebar.selectbox(
    "Fuel_Type",
    [
        'Petrol',
        'Diesel',
        'CNG',
        'Electric'
    ]
)


Transmission = st.sidebar.selectbox(
    "Transmission",
    [
        'Manual',
        'Automatic'
    ]
)


Previous_Owners = st.sidebar.number_input(
    "Previous_Owners",
    min_value=0,
    value=2
)


Seats = st.sidebar.number_input(
    "Seats",
    min_value=1,
    value=7
)


Location = st.sidebar.selectbox(
    "Location",
    [
        'Delhi',
        'Pune',
        'Kolkata',
        'Hyderabad',
        'Chennai',
        'Durgapur',
        'Bengaluru',
        'Mumbai'
    ]
)


# ============================================================
# CHAT STATE
# ============================================================

class ChatState(TypedDict, total=False):

    # Conversation
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    question: str

    # Car information
    Brand: str
    Car_Age: float
    Kilometers_Driven: float
    Engine_CC: float
    Mileage_KMPL: float
    Fuel_Type: str
    Transmission: str
    Previous_Owners: int
    Seats: int
    Location: str

    # ML prediction
    prediction: float

    # Purchase research
    is_purches: str

    # Final response
    final_answer: str

    # Validation
    is_valid: bool
    missing_fields: list[str]


# ============================================================
# UNDERSTAND INPUT NODE
# ============================================================

def understand_input(state: ChatState):

    question = state["question"]

    prompt = f"""
You are a car information extraction system.

Extract the following information from the user's input.

Return ONLY the values separated by commas.

The order MUST be exactly:

Brand,
Car_Age,
Kilometers_Driven,
Engine_CC,
Mileage_KMPL,
Fuel_Type,
Transmission,
Previous_Owners,
Seats,
Location

Rules:

1. Return exactly 10 values.
2. Do not add labels.
3. Do not add explanations.
4. Do not use markdown.
5. Preserve the values given by the user.
6. If a value is missing, return 0.
7. Brand must be a string.
8. Fuel_Type must be a string.
9. Transmission must be a string such as Manual or Automatic.
10. Location must be a string.
11. Car_Age must be numerical.
12. Kilometers_Driven must be numerical.
13. Engine_CC must be numerical.
14. Mileage_KMPL must be numerical.
15. Previous_Owners must be numerical.
16. Seats must be numerical.
17. Never convert Manual/Automatic into 0/1.

Example:

User:
What is the car price of a Brand 'Hyundai',
car age 9,
kilometers driven 31788,
engine_cc 1498,
mileage 20.7,
fuel type 'Petrol',
transmission 'Manual',
previous owner 2,
seats 5,
Location 'Delhi'

Output:

Hyundai, 9, 31788, 1498, 20.7, Petrol, Manual, 2, 5, Delhi

User input:

{question}
"""

    response = model.invoke(prompt)

    print("\n==============================")
    print("LLM EXTRACTION")
    print("==============================")

    print(response.content)

    response_text = response.content.strip()

    response_text = (
        response_text
        .replace("```", "")
        .strip()
    )

    values = [
        v.strip()
        for v in response_text.split(",")
    ]

    while len(values) < 10:
        values.append("0")

    values = values[:10]

    return {

        "Brand": str(values[0]),

        "Car_Age": float(values[1]),

        "Kilometers_Driven": float(values[2]),

        "Engine_CC": float(values[3]),

        "Mileage_KMPL": float(values[4]),

        "Fuel_Type": str(values[5]),

        "Transmission": str(values[6]),

        "Previous_Owners": int(float(values[7])),

        "Seats": int(float(values[8])),

        "Location": str(values[9])

    }


# ============================================================
# PREDICTION NODE
# ============================================================

def prediction_node(state: ChatState):

    input_data = {

        "Brand": state["Brand"],

        "Car_Age": state["Car_Age"],

        "Kilometers_Driven": state["Kilometers_Driven"],

        "Engine_CC": state["Engine_CC"],

        "Mileage_KMPL": state["Mileage_KMPL"],

        "Fuel_Type": state["Fuel_Type"],

        "Transmission": state["Transmission"],

        "Previous_Owners": state["Previous_Owners"],

        "Seats": state["Seats"],

        "Location": state["Location"]

    }

    response = requests.post(
        api_url,
        json=input_data,
        timeout=60
    )

    if response.status_code != 200:

        raise Exception(
            f"Prediction API Error: "
            f"{response.status_code} - "
            f"{response.text}"
        )

    result = response.json()

    prediction = float(
        result["predicted_category"]
    )

    print("\n==============================")
    print("ML PREDICTION API")
    print("==============================")

    print(
        f"Predicted Price: ₹{prediction:,.2f}"
    )

    return {
        "prediction": prediction
    }


# ============================================================
# DUCKDUCKGO SEARCH
# ============================================================

search = DuckDuckGoSearchRun()


@tool
def purchase_or_not(
    brand: str,
    car_age: float,
    kilometers_driven: float,
    engine_cc: float
):

    """
    Research a used car and assess whether it may be a good
    second-hand purchase.
    """

    prompt = f"""
Research this used car:

Brand: {brand}

Car Age: {car_age} years

Kilometers Driven: {kilometers_driven} km

Engine CC: {engine_cc} cc


Find information about:

1. Brand reliability
2. Common problems
3. Maintenance cost
4. Engine reliability
5. Expected engine life
6. Common issues for older vehicles
7. Whether this type of used car is generally a good purchase


Give a concise research summary.

Do not make up specific facts if reliable information is unavailable.
"""

    result = search.invoke(prompt)

    return result


# ============================================================
# PURCHASE PREDICTION NODE
# ============================================================

def purchase_prediction(state: ChatState):

    print("\n==============================")
    print("USED CAR RESEARCH")
    print("==============================")

    result = purchase_or_not.invoke({

        "brand": state["Brand"],

        "car_age": state["Car_Age"],

        "kilometers_driven":
            state["Kilometers_Driven"],

        "engine_cc":
            state["Engine_CC"]

    })

    print(result)

    return {

        "is_purches": result

    }


# ============================================================
# FINAL NODE
# ============================================================

def final_node(state: ChatState):

    prediction = float(
        state["prediction"]
    )

    prompt = f"""
You are an experienced used-car advisor.

Analyze the following car using BOTH:

1. Machine Learning predicted price
2. Online used-car research

CAR DETAILS
===========

Brand:
{state["Brand"]}

Car Age:
{state["Car_Age"]} years

Kilometers Driven:
{state["Kilometers_Driven"]} km

Engine CC:
{state["Engine_CC"]} cc

Mileage:
{state["Mileage_KMPL"]} km/l

Fuel Type:
{state["Fuel_Type"]}

Transmission:
{state["Transmission"]}

Previous Owners:
{state["Previous_Owners"]}

Seats:
{state["Seats"]}

Location:
{state["Location"]}


MACHINE LEARNING PREDICTION
===========================

Predicted Price:
₹{state["prediction"]:,.2f}


USED-CAR ONLINE RESEARCH
========================

{state["is_purches"]}


Prepare a useful final used-car assessment.

Include:

1. Estimated market price
2. Car reliability
3. Important problems to check
4. Maintenance considerations
5. Impact of car age
6. Impact of kilometers driven
7. Whether the engine specification is suitable
8. Whether this appears to be a good second-hand purchase
9. What price would be reasonable
10. Final recommendation

Use one of these final recommendations:

BUY
CONSIDER
AVOID

Important:
The ML price is an estimate, not a guaranteed market price.
The online research may not describe the exact vehicle.
The buyer should inspect the actual vehicle and verify its
service history, accident history, ownership documents and
mechanical condition before purchasing.

Format the response clearly using Markdown.
"""

    response = model.invoke(prompt)

    print(
        f"Predicted Price: ₹{prediction:,.2f}"
    )

    return {

        "final_answer": response.content,

        "prediction": prediction

    }


# ============================================================
# LANGGRAPH
# ============================================================

graph = StateGraph(ChatState)


graph.add_node(
    "understand_input",
    understand_input
)


graph.add_node(
    "prediction_node",
    prediction_node
)


graph.add_node(
    "purchase_prediction",
    purchase_prediction
)


graph.add_node(
    "final_node",
    final_node
)


graph.add_edge(
    START,
    "understand_input"
)


graph.add_edge(
    "understand_input",
    "prediction_node"
)


graph.add_edge(
    "prediction_node",
    "purchase_prediction"
)


graph.add_edge(
    "purchase_prediction",
    "final_node"
)


graph.add_edge(
    "final_node",
    END
)


app = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.sidebar.button(
    "🚗 Predict Car Price",
    use_container_width=True
):

    # ========================================================
    # CREATE QUESTION FOR YOUR LLM EXTRACTION NODE
    # ========================================================

    question = f"""
What is the car price of a Brand '{Brand}',
car age {Car_Age},
kilometers driven {Kilometers_Driven},
engine_cc {Engine_CC},
mileage {Mileage_KMPL},
fuel type '{Fuel_Type}',
transmission '{Transmission}',
previous owner {Previous_Owners},
seats {Seats},
Location '{Location}'
"""


    # ========================================================
    # SHOW INPUT
    # ========================================================

    st.subheader("🚗 Selected Car Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(f"**Brand:** {Brand}")

        st.write(
            f"**Car Age:** {Car_Age} years"
        )

        st.write(
            f"**Kilometers:** "
            f"{Kilometers_Driven:,} km"
        )

        st.write(
            f"**Engine:** "
            f"{Engine_CC:,} cc"
        )


    with col2:

        st.write(
            f"**Mileage:** "
            f"{Mileage_KMPL} km/l"
        )

        st.write(
            f"**Fuel:** {Fuel_Type}"
        )

        st.write(
            f"**Transmission:** "
            f"{Transmission}"
        )


    with col3:

        st.write(
            f"**Previous Owners:** "
            f"{Previous_Owners}"
        )

        st.write(
            f"**Seats:** {Seats}"
        )

        st.write(
            f"**Location:** {Location}"
        )


    # ========================================================
    # RUN LANGGRAPH
    # ========================================================

    with st.spinner(
        "🤖 AI is analyzing the car..."
    ):

        try:

            result = app.invoke(

                {
                    "question": question
                },

                config={
                    "configurable": {
                        "thread_id":
                            "streamlit_car_analysis"
                    }
                }

            )


            # =================================================
            # ML PREDICTION
            # =================================================

            st.divider()

            st.subheader(
                "💰 Machine Learning Prediction"
            )

            prediction = result["prediction"]


            st.metric(
                "Predicted Used-Car Price",
                f"₹{prediction:,.2f}"
            )


            # =================================================
            # WEB RESEARCH
            # =================================================

            st.divider()

            st.subheader(
                "🌐 Used-Car Research"
            )

            with st.expander(
                "View Online Research"
            ):

                st.write(
                    result["is_purches"]
                )


            # =================================================
            # FINAL REPORT
            # =================================================

            st.divider()

            st.subheader(
                "🤖 Final Used-Car Assessment"
            )

            st.markdown(
                result["final_answer"]
            )


        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to the FastAPI "
                "prediction service."
            )


        except requests.exceptions.Timeout:

            st.error(
                "⏳ The FastAPI prediction service "
                "timed out. Please try again."
            )


        except Exception as e:

            st.error(
                "❌ An error occurred while running "
                "the LangGraph application."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Used Car Advisor | "
    "Streamlit + LangGraph + Groq + FastAPI + "
    "Machine Learning Regression + DuckDuckGo | Tathagata Nath"
)
