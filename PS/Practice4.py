import os;

# print(os.getcwd())
# print(os.chdir("E:\Coding\Python"))
print(os.listdir(path='./PS'))

# os.mkdir("E:\Coding\Python\Practice",mode=0o777,dir_fd=None)
# os.rmdir("./Practice",dir_fd=None)
# os.chdir("E:\Coding\Python")

# os.rename("first.py","second.py")

# print(os.listdir())

# showing files of giving directory
contents=os.listdir(path='./PS');

# printing files
for items in contents:
    print(items);
