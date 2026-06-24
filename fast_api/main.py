from fastapi import FastAPI

app = FastAPI(
    title="My FastAPI Application",
    description="This gives the idea of basic FastAPI functionality.",
    version="1.0.0",
    contact={
        "name": "Shailendra Singh Rawat",
        "email": "shailendra.singh.rawat@example.com"
    }
)

# create a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}
