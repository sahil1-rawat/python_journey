from pydantic import BaseModel

class Product(BaseModel):
    id:int
    name:str
    price :float
    in_stock:bool


input_data={
    "id": '101',
    "name": "Laptop",
    "price": 999,
    "in_stock": 'true'
 }   

product=Product(**input_data);
print(product)