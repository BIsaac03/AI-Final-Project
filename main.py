print("Welcome to the language detector")
print("The following languages are supported:\n")
print("1. English")
print("2. Spanish")
print("3. French")
print("4. Portuguese")
print("5. Indonesian")
print("6. German")
print("7. Turkish")
print("8. Vietnamese")
print("9. Italian")
print("10. Hungarian")

keepGoing = True

while keepGoing:
    word = input("To view more details about the previous entry, enter '+'. To exit the program, enter 'EXIT'.  ")

    # quits program
    if word == 'EXIT':
        keepGoing = False

    # expands on details of previous word
    elif word == '+':
        print("ADD DETAILS")
        
    # analyzes new word
    else:
        print("ANALYZE WORD")