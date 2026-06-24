import base64 #using for encoding
import json
import hmac
import hashlib 

header={"alg":"HS256","typ":"JWT"}
payload={"user_id":123, "username":"shailendra"}

secret="my_secret_key_sh"

def base64url_encode(data):
    json_str=json.dumps(data,separators=(',',':')).encode()
    return base64.urlsafe_b64encode(json_str).rstrip(b'=').decode()

encode_header=base64url_encode(header)
encode_payload=base64url_encode(payload)
# signature_input=f"{encode_header}.{encode_payload}".encode()

print("Encoded Header:", encode_header)
print("Encoded Payload:", encode_payload)

# creating signature
signature=hmac.new(key=secret.encode(), msg=f"{encode_header}.{encode_payload}".encode(), digestmod=hashlib.sha256).digest()

encoded_signature=base64.urlsafe_b64encode(signature).rstrip(b'=').decode()

print("Signature:", signature)
print("Encoded Signature:", encoded_signature)

# final JWT token
jwt_token=f"{encode_header}.{encode_payload}.{encoded_signature}"
print("Final JWT Token:", jwt_token)
