import json
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from graph.graphinitializer import Initializer
from langchain_core.messages import HumanMessage


app = FastAPI()

session_data = {}

graph = Initializer()

# Function to retrieve or create memory for a session

@app.get("/")
def read_root():
    return {"message": "Hello, welcome to the multi-user API"}

# Endpoint to create a session and return session ID
@app.post("/create_session")
async def create_session():
    user_id = str(uuid.uuid4())  # Generate a unique user/session ID
    
    return {"message": "Session created successfully!", "session_id": user_id}

@app.post("/upload_data/{user_id}")
async def store_data(request: Request, user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="Session ID not provided")

    # Store the incoming data in the session-specific data store
    data = await request.json()
    session_data[user_id] = data
    return JSONResponse(content={"Message": "Data uploaded and processed successfully!"}, status_code=200)

@app.post("/chat/{user_id}")
async def chat(request: Request, user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="Session ID not provided")


    data = await request.json()
    query = data.get('query')
    
    if not query:
        raise HTTPException(status_code=400, detail="Query not provided")
    
    config =  {"configurable": {"thread_id": user_id}}

    messages = [HumanMessage(content=query)]
    messages = graph.invoke({"messages": messages,"dataset":session_data[user_id]},config,stream_mode="values")
    response = messages['messages'][-1].content
    tool_type = messages['messages'][-2].additional_kwargs['tool_calls'][0]['function']['name'].split('_')[0]
    return {"response": response,"type":tool_type}
    
    # Use the classifier chain to determine the response type
    """ response_type = session_data['classifier_chain'].run(query).strip().lower()
    
    if "text" in response_type:
        response = session_data['text_chain'].run(query=query, dataset=session_data['stored_data'])
        return {"response": response, "type": "text"}
    
    elif "chart" in response_type:
        chart_specs = session_data['chart_chain'].run(query=query, dataset=session_data['stored_data'])
        return {"chart": chart_specs, "type": "chart"}
    
    else:
        raise HTTPException(status_code=400, detail=f"Unable to determine response type: {response_type}") """

# Clear history endpoint using user_id
@app.post("/delete_session_data/{user_id}")
async def delete_session_data(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="Session ID not provided")
    
    if user_id in session_data:
        # Clear the session memory and stored data
        session_data.pop(user_id, None)
        return {"message": "Chat history and session data cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="debug")
