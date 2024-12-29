import os
import dotenv
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from pymongo import MongoClient
from bson import ObjectId
from langchain_core.runnables.config import RunnableConfig
from datetime import datetime


class AgentState(TypedDict):
    question: str
    mongo_query: str
    query_result: str
    query_rows: list
    current_user: str
    attempts: int
    relevance: str
    mongo_error: bool
    final_response: str



dotenv.load_dotenv()
mongo_uri = os.environ['MONGO_URI']
schema_name = os.environ['SCHEMA_NAME']
user_collection= os.environ['USER_COLLECTION']
client = MongoClient(mongo_uri)

class GetCurrentUser(BaseModel):
    current_user: str = Field(
        description="The name of the current user based on the provided user ID."
    )

class ConvertToMongoQuery(BaseModel):
    mongo_query: str = Field(
        description="The MongoDB query corresponding to the user's natural language question."
    )

class CheckRelevance(BaseModel):
    relevance: str = Field(
        description="Indicates whether the question is related to the database schema. 'relevant' or 'not_relevant'."
    )

def smart_dict_parser(input_string: str):
    """
    prende in input una stringa formattata come dizionario e la restituisce con doppie parentesi graffe.
    
    Args:
        input_dict (str): Dizionario di input da trasformare.
    
    Returns:
        str: Stringa JSON con doppie parentesi graffe.
    """
    # Raddoppia le parentesi graffe
    return input_string.replace('{', '{{').replace('}', '}}')
    
     

def analyze_field(value):
    """
    Analizza il tipo del campo e ritorna una descrizione.
    """
    field_type = type(value).__name__
    
    if isinstance(value, dict):
        return "Documento annidato"
    elif isinstance(value, list):
        if value:
            first_item_type = type(value[0]).__name__
            return f"Lista di {first_item_type}"
        else:
            return "Lista vuota"
    else:
        return field_type
    

def analyze_nested_document(nested_doc, indent=2):
    """
    Analizza ricorsivamente la struttura di un documento annidato.
    """
    nested_schema = ""
    for field, value in nested_doc.items():
        nested_field_type = analyze_field(value)
        
        nested_schema += " " * indent + f"- {field}: {nested_field_type}\n"
        
        if isinstance(value, dict):
            nested_schema += analyze_nested_document(value, indent + 2)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            nested_schema += " " * (indent + 2) + f"Struttura della lista:\n"
            nested_schema += analyze_nested_document(value[0], indent + 4)
    
    return nested_schema

def get_mongodb_schema(collection_name):
    """
    Ritorna le informazioni della collezione passata come input.
    Restituisce il nome della collezione con il tipo di dato per ogni campo della collezione.
    Questo mapping viene fatto prendendo l'ultimo document inserito nella collezione.
    """
    client = MongoClient(mongo_uri)
    db = client[schema_name]
    collection = db[collection_name]

    sample_doc = collection.find_one()
    if not sample_doc:
        print(f"Nessun documento trovato nella collezione {collection_name}")
        return "Collezione vuota"
    
    schema = f"Collezione: {collection_name}\n"
    
    for field, value in sample_doc.items():
        if field == '_id':
            schema += f"- _id: ObjectId (Chiave primaria)\n"
        else:
            field_type = analyze_field(value)
            schema += f"- {field}: {field_type}\n"
            
            if isinstance(value, dict):
                schema += analyze_nested_document(value, indent=2)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                schema += "  Struttura della lista:\n"
                schema += analyze_nested_document(value[0], indent=4)
    
    schema += f"\nNumero totale di documenti: {collection.count_documents({})}\n"
    print(f"Schema recuperato per la collezione {collection_name}")
    return schema


def get_current_user(state: dict, config: RunnableConfig):
    print("Retrieving the current user based on user ID.")
    
    # Seleziona il database
    db = client[schema_name]
    users_collection = db[user_collection] 
    
    # Ottieni l'ID utente dalla configurazione
    user_id = config.get("configurable", {}).get("current_user_id", None)
    if not user_id:
        state["current_user"] = "User not found"
        print("No user ID provided in the configuration.")
        return state

    try:
        user_id = ObjectId(user_id) if isinstance(user_id, str) else user_id
        
        user = users_collection.find_one({"_id": user_id})
        
        if user:
            state["current_user"] = user.get('username', 'User without username')
            print(f"Current user set to: {state['current_user']}")
        else:
            state["current_user"] = "User not found"
            print("User not found in the database.")
    
    except Exception as e:
        state["current_user"] = "Error retrieving user"
        print(f"Error retrieving user: {str(e)}")
    
    finally:

        client.close()
    
    return state

def check_relevance(state: dict, config: RunnableConfig):

    question = state["question"]
    
    schema = get_mongodb_schema("Spesa")
    
    print(f"Checking relevance of the question: {question}")
    
    system = """You are an assistant that determines whether a given question is related to the following database schema.  Schema: {schema}  Respond with only "relevant" or "not_relevant". """.format(schema=schema)
    
    human = f"Question: {question}"
    
    check_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    
    dotenv.load_dotenv()
    llm = ChatOpenAI(temperature=0)
    structured_llm = llm.with_structured_output(CheckRelevance)
    
    relevance_checker = check_prompt | structured_llm
    
    try:
        relevance = relevance_checker.invoke({})
        state["relevance"] = relevance.relevance
        print(f"Relevance determined: {state['relevance']}")
    except Exception as e:
        print(f"Error checking relevance: {str(e)}")
        state["relevance"] = "not_relevant"
    
    
    return state

def convert_nl_to_mongo(state: dict, config: RunnableConfig):

    question = state["question"]
    current_user = state["current_user"]
    
    # Genera lo schema del database
    schema = get_mongodb_schema('Spesa')
    
    print(f"Converting question to MongoDB query for user '{current_user}': {question}")
    
    system = """Sei un assistente che genera query di aggregazione per MongoDB basate su input dell'utente. La collezione MongoDB ha il seguente schema descritto:
            {schema}


            Lo user corrente è '{current_user}'. Assicurati che tutte le query riguardino sono l'utente corrente.
            Il giorno corrente è {current_day}. Assicurati che le query temporali usino questo giorno come riferimento.


            Regole:

            -Usa la notazione a punti per i documenti nidificati
            -Usa $eq, $gt, $lt per i confronti
            -Assicurati che le query siano specifiche per l'utente

            Fornisci la query in questo formato:
            {{
                "filter": {{}},
                "projection": {{}},
                "sort": {{}}
            }}

            Rispondi solo generando la query nel formato indicato.
            User Question: {question}
            Query:
"""
    
    convert_prompt = PromptTemplate(
        input_variables=["schema", "current_user", "question", "current_day"],
        template=system
    )
    
    dotenv.load_dotenv()
    llm = ChatOpenAI(temperature=0)
    structured_llm = llm.with_structured_output(ConvertToMongoQuery)
    
    mongo_generator = convert_prompt | structured_llm
    inputs= {
        "schema":schema,
        "current_user":current_user,
        "question": question,
        "current_day" : datetime.now()
    }
    
    try:
        result = mongo_generator.invoke(inputs)
        state["mongo_query"] = result.mongo_query
        print(f"Generated MongoDB query: {state['mongo_query']}")
    except Exception as e:
        print(f"Error generating MongoDB query: {str(e)}")
        state["mongo_query"] = "{}"

    
    return state

def execute_mongo_query(state: dict):
    client = MongoClient(mongo_uri)
    
    db = client[schema_name]
    
    mongo_query_str = state["mongo_query"].strip()
    print(f"Executing MongoDB query: {mongo_query_str}")
    
    try:
        mongo_query = eval(mongo_query_str)
        
        collection_name = "Spesa"
        filter_query = mongo_query.get("filter", {})
        projection = mongo_query.get("projection", {})
        sort = mongo_query.get("sort", None)
        
        collection = db[collection_name]
        
        if sort:
            cursor = collection.find(filter_query, projection).sort(list(sort.items()))
        else:
            cursor = collection.find(filter_query, projection)
        
        results = list(cursor)
        
        if results:
            # Converti ObjectId in stringhe per la serializzazione
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            state["query_rows"] = results
            
            # Formatta il risultato in modo leggibile
            formatted_results = []
            for result in results:
                # Personalizza la formattazione in base ai campi specifici della tua collezione
                formatted_result = ", ".join([f"{k}: {v}" for k, v in result.items()])
                formatted_results.append(formatted_result)
            
            state["query_result"] = "\n".join(formatted_results)
            state["query_result"] = smart_dict_parser(state["query_result"])
            state["mongo_error"] = False
            print("MongoDB query executed successfully.")
        else:
            state["query_rows"] = []
            state["query_result"] = "No results found."
            state["mongo_error"] = False
    
    except Exception as e:
        state["query_result"] = f"Error executing MongoDB query: {str(e)}"
        state["mongo_error"] = True
        print(f"Error executing MongoDB query: {str(e)}")
    
    finally:
        client.close()
    
    return state


def generate_human_readable_answer(state: dict):
    mongo_query = smart_dict_parser(state.get("mongo_query", ""))
    print(f"La query: {mongo_query}")
    result = state.get("query_result", "")
    print(f"risultato: {result}")

    current_user = state.get("current_user", "User")
    query_rows = state.get("query_rows", [])
    mongo_error = state.get("mongo_error", False)
    question = state.get("question","")

    print("Generating a human-readable answer.")
    
    
    inputs={
        "current_user":current_user,
        "mongo_query": mongo_query,
        "result":result,
        "question": question
    }

    if mongo_error:
        mongo_error_prompt = """
        Sei un assistente virtuale che fa analisi delle spese di un utente dentro un app di tracciamento delle spese, ma lo fai con un tono simpatico e amichevole.
        L'analisi ha riscontrato un problema e bisogna comunicarlo all'utente.
            \nUser's Question: 
            {question}

            \nMongoDB Query:
            {mongo_query}

            Result:
            {result}
            Formula un messaggio di errore chiaro e comprensibile per un utente non tecnico in una sola frase, inizia con 'Ciao {current_user},' e informalo sul problema """
        generate_prompt= PromptTemplate(
        input_variables=["mongo_query", "current_user", "result","question"],
        template=mongo_error_prompt
    )
                                    
    
    elif not query_rows:
        # Nessun risultato trovato
        no_query_rows_template = """
        Sei un assistente virtuale che fa analisi delle spese di un utente dentro un app di tracciamento delle spese, ma lo fai con un tono simpatico e amichevole.
        L'analisi effettuata sui dati dell'utente non ha prodotto risultati e bisogna comunicarglielo.       
        \nUser's Question: 
        {question}

        \nMongoDB Query:
        {mongo_query}

        Result:
        {result}

        Formula una chiara e comprensibile risposta alla domanda originale dell'utente in una sola frase. Inizia con 'Ciao {current_user},' e menziona il fatto che l'analisi non ha prodotto risultati."""   
        generate_prompt= PromptTemplate(
        input_variables=["mongo_query", "current_user", "result","question"],
        template=no_query_rows_template
    ) 
    else:

        result_template="""
        Sei un assistente virtuale che fa analisi delle spese di un utente dentro un app di tracciamento delle spese, ma lo fai con un tono simpatico e amichevole.
        ti verrà fornito il risultato di una query su un database di MongoDB contenente le spese dell'utente che effettua la domanda, analizza il dato e rispondi alla domanda dell'utente.
        Sii molto analitico, se ti viene richiesto individua trand di spesa o comportamenti ricorrenti, dai peso ad ogni singolo dato.
        \nUser's Question: 
        {question}

        \nMongoDB Query:
        {mongo_query}

        Result:
        {result}

        Rispondi in modo discorsivo, simpatico e utilizza emoji quando appropriato! 🎉😊
        Inizia la risposta con 'Ciao {current_user},'.
        Limitati ad analizzare i dati e dare risposte utili, non proporre di fare cose all'utente.
        Ricorda che l'unità di misura riguardante le spese sono gli Euro (€)."""

        generate_prompt= PromptTemplate(
        input_variables=["mongo_query", "current_user", "result","question"],
        template=result_template
    ) 
    
    dotenv.load_dotenv()
    llm = ChatOpenAI(temperature=0)
    human_response = generate_prompt | llm 
    
    try:
        answer = human_response.invoke(inputs)
        state["final_response"] = answer.content
        print("Generated human-readable answer.")
    except Exception as e:
        state["query_result"] = f"Error generating response: {str(e)}"
        print(f"Error generating human-readable answer: {str(e)}")
    
    return state

class RewrittenQuestion(BaseModel):
    question: str = Field(description="The rewritten question.")

def regenerate_query(state: dict):
    question = state["question"]
    print("Regenerating the MongoDB query by rewriting the question.")
    
    system = """You are an assistant that reformulates an original question to enable more precise MongoDB queries. Ensure that all necessary details are preserved to retrieve complete and accurate data.
    
    Consider:
    - Specifying precise filter conditions
    - Including relevant projection fields
    - Handling nested document structures
    """
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", f"Original Question: {question}\n\nReformulate the question to enable more precise MongoDB queries, ensuring all necessary details are preserved."),
    ])
    
    dotenv.load_dotenv()
    llm = ChatOpenAI(temperature=0)
    structured_llm = llm.with_structured_output(RewrittenQuestion)
    rewriter = rewrite_prompt | structured_llm
    
    try:
        rewritten = rewriter.invoke({})
        state["question"] = rewritten.question
        state["attempts"] = state.get("attempts", 0) + 1
        print(f"Rewritten question: {state['question']}")
    except Exception as e:
        print(f"Error regenerating query: {str(e)}")
        # Mantieni la domanda originale in caso di errore
        state["attempts"] = state.get("attempts", 0) + 1
    
    return state

def generate_funny_response(state: dict):
    print("Generating a funny response for an unrelated question.")
    
    system = """You are a charming and funny assistant who helps manage personal expenses. 
                you have to respond in a playful manner to questions that cannot be answered by database queries. Be creative, witty, and slightly expenses-related."""
    
    question = state["question"]
    human = f"Question: {question}"
    
    funny_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    
    llm = ChatOpenAI(temperature=0.7)
    funny_response = funny_prompt | llm | StrOutputParser()
    
    try:
        message = funny_response.invoke({})
        state["final_response"] = message
        print("Generated funny response.")
    except Exception as e:
        state["final_response"] = "Oops! My humor chip seems to be malfunctioning. How about a pizza?"
        print(f"Error generating funny response: {str(e)}")
    
    return state


def end_max_iterations(state: AgentState):
    state["query_result"] = "Please try again."
    print("Maximum attempts reached. Ending the workflow.")
    return state

def relevance_router(state: AgentState):
    if state["relevance"].lower() == "relevant":
        return "convert_to_mongo"
    else:
        return "generate_funny_response"

def check_attempts_router(state: AgentState):
    if state["attempts"] < 3:
        return "convert_to_mongo"
    else:
        return "end_max_iterations"

def execute_mongo_router(state: AgentState):
    if not state.get("mongo_error", False):
        return "generate_human_readable_answer"
    else:
        return "regenerate_query"

workflow = StateGraph(AgentState)

workflow.add_node("get_current_user", get_current_user)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("convert_to_mongo", convert_nl_to_mongo)
workflow.add_node("execute_mongo", execute_mongo_query)
workflow.add_node("generate_human_readable_answer", generate_human_readable_answer)
workflow.add_node("regenerate_query", regenerate_query)
workflow.add_node("generate_funny_response", generate_funny_response)
workflow.add_node("end_max_iterations", end_max_iterations)

workflow.add_edge("get_current_user", "check_relevance")

workflow.add_conditional_edges(
    "check_relevance",
    relevance_router,
    {
        "convert_to_mongo": "convert_to_mongo",
        "generate_funny_response": "generate_funny_response",
    },
)

workflow.add_edge("convert_to_mongo", "execute_mongo")

workflow.add_conditional_edges(
    "execute_mongo",
    execute_mongo_router,
    {
        "generate_human_readable_answer": "generate_human_readable_answer",
        "regenerate_query": "regenerate_query",
    },
)

workflow.add_conditional_edges(
    "regenerate_query",
    check_attempts_router,
    {
        "convert_to_mongo": "convert_to_mongo",
        "max_iterations": "end_max_iterations",
    },
)

workflow.add_edge("generate_human_readable_answer", END)
workflow.add_edge("generate_funny_response", END)
workflow.add_edge("end_max_iterations", END)

workflow.set_entry_point("get_current_user")

app = workflow.compile()

def main(user_id):
    """
    esempio di utilizzo
    """
    user_question_1 = "Indicami la categoria in cui ho speso di nell'ultimo mese"
    fake_config = {"configurable": {"current_user_id": user_id}}
    result_1 = app.invoke({"question": user_question_1, "attempts": 0}, config=fake_config)
    print(result_1["final_response"])