import time

# Get the current time
hrs = int(time.strftime("%H"))
ampm_indicator = time.strftime("%p", time.localtime())

# Check the time of day and print appropriate greeting
if hrs >= 4 and hrs < 12 :
    print("Good Morning")
elif hrs >= 12 and hrs < 16 :  # 12 PM to 4 PM
    print("Good Afternoon")
elif hrs >= 16 and hrs < 22 :  # 4 PM to 10 PM
    print("Good Evening")
else:
    print('Good Night')
