from pydantic import BaseModel
class User(BaseModel):
    id:int
    name:str
    is_active:bool


input_data={
    "id": '1',
    "name": "John Doe",
    "is_active": 'true'
}

user=User(**input_data)
print(user)