import secrets
import hmac
import hashlib


# server secret key (should be kept secure and not hardcoded in production)
SERVER_SECRET= b"<SERVER_SECRET>"

# step 1: create API key for client
def create_api_key():
    return 'kp_'+secrets.token_urlsafe(32)


# step 2: convert API key to hash for database storage
def hash_api_key(api_key: str):
    return hmac.new(SERVER_SECRET, api_key.encode(), hashlib.sha256).hexdigest()

# step 3: verify API key during request
def verify_api_key(api_key: str, stored_hash: str):
    return hash_api_key(api_key) == stored_hash

# create a new API key and hash it for storage
new_api_key = create_api_key()
hashed_key = hash_api_key(new_api_key)
print("Generated API Key:", new_api_key)
print("Hashed API Key for Storage:", hashed_key)

# verifying the API key
is_valid = verify_api_key(new_api_key, hashed_key)
print("Is the API Key valid?", is_valid)